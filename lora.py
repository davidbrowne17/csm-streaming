import json
import math
import os
import glob
import torch
import torchaudio
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from pathlib import Path
from transformers import AutoTokenizer, get_scheduler
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from safetensors.torch import save_file
import csv
from models import Model
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from tokenizers.processors import TemplateProcessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("finetune.log")]
)
logger = logging.getLogger(__name__)

AUDIO_DIR = "audio_data"
OUTPUT_DIR = "finetuned_model"
NUM_EPOCHS = 15
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 0.1
USE_WANDB = False
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = True
WARMUP_STEPS = 50
SPEAKER_ID = 0
MODEL_NAME = "sesame/csm-1b"
TRANSCRIPTION_MODEL = "openai/whisper-large-v3-turbo"
MAX_AUDIO_FILES = 0
R = 32
ALPHA = 32
VALIDATION_FRACTION = 0.1
DECODER_CHUNK_SIZE = 32
MAX_SEQ_LEN = 2048

from finetuning import (
    TARGET_LAYERS, LoRALinear, replace_linear_with_lora, merge_lora_weights,
    remove_lora_modules, forward_and_loss, calculate_validation_loss,
    save_checkpoint, split_dataset, latest_training_checkpoint, load_checkpoint,
)

class TrainingVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.epochs = []
        self.losses = []
        self.val_losses = []  # Added validation losses
        self.learning_rates = []
        self.steps = []
        
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 15))
        self.fig.suptitle('CSM Finetuning Progress', fontsize=16)
        
        # Setup training loss plot
        self.axes[0].set_title('Training Loss')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Setup validation loss plot
        self.axes[1].set_title('Training vs Validation Loss')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Loss')
        self.axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Setup learning rate plot
        self.axes[2].set_title('Learning Rate')
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('Learning Rate')
        self.axes[2].grid(True, linestyle='--', alpha=0.7)
    
    def update(self, epoch, step, loss, lr, val_loss=None):
        """Update the metrics and redraw the plot"""
        self.epochs.append(epoch)
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        
        # Add validation loss if provided, otherwise use None
        if val_loss is not None:
            self.val_losses.append(val_loss)
        elif len(self.val_losses) > 0:
            # If we have validation losses but none provided this time, use the last one
            self.val_losses.append(self.val_losses[-1])
        else:
            # If we've never had validation losses, use None
            self.val_losses.append(None)
        
        # Update training loss plot
        self.axes[0].clear()
        self.axes[0].plot(self.epochs, self.losses, 'b-')
        self.axes[0].set_title('Training Loss')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True, linestyle='--', alpha=0.7)
        
        # Update validation loss plot
        self.axes[1].clear()
        self.axes[1].plot(self.epochs, self.losses, 'b-', label='Training')
        
        # If we have validation losses, plot them
        if any(x is not None for x in self.val_losses):
            # Filter out None values
            val_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
            val_loss_values = [v for v in self.val_losses if v is not None]
            if val_epochs:
                self.axes[1].plot(val_epochs, val_loss_values, 'r-', label='Validation')
                self.axes[1].legend()
        
        self.axes[1].set_title('Training vs Validation Loss')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Loss')
        self.axes[1].grid(True, linestyle='--', alpha=0.7)
        
        # Update learning rate plot
        self.axes[2].clear()
        self.axes[2].plot(self.epochs, self.learning_rates, 'g-')
        self.axes[2].set_title('Learning Rate')
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('Learning Rate')
        self.axes[2].grid(True, linestyle='--', alpha=0.7)
        
        # Calculate convergence metrics
        min_loss = min(self.losses)
        min_loss_epoch = self.epochs[self.losses.index(min_loss)]
        
        # Check for potential convergence stall
        recent_window = 10  # Look at last 10 steps
        if len(self.losses) > recent_window:
            recent_losses = self.losses[-recent_window:]
            loss_std = np.std(recent_losses)
            loss_change = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] if recent_losses[0] != 0 else 0
            
            convergence_status = ""
            if loss_std < 0.001 and loss_change < 0.01:
                convergence_status = "STALLED: Loss not improving significantly"
            elif min_loss == self.losses[-1]:
                convergence_status = "IMPROVING: New best loss!"
            elif self.losses[-1] < self.losses[-2]:
                convergence_status = "IMPROVING: Loss decreasing"
            else:
                convergence_status = "FLUCTUATING: Loss increased"
            
            # Add convergence status to title
            self.fig.suptitle(f'CSM Finetuning Progress - {convergence_status}\n' + 
                            f'Epoch: {epoch:.2f}, Loss: {loss:.4f}, LR: {lr:.8f}\n' + 
                            f'Best: {min_loss:.4f} at epoch {min_loss_epoch:.2f}', fontsize=12)
        else:
            self.fig.suptitle(f'CSM Finetuning Progress\n' + 
                            f'Epoch: {epoch:.2f}, Loss: {loss:.4f}, LR: {lr:.8f}\n' + 
                            f'Best: {min_loss:.4f} at epoch {min_loss_epoch:.2f}', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust for the larger title
        
        # Save the figure
        plot_path = os.path.join(self.output_dir, 'training_progress.png')
        self.fig.savefig(plot_path)
        
    def finalize(self):
        """Create a final, more detailed visualization when training completes"""
        # Create a new figure for the final plot
        final_fig = plt.figure(figsize=(12, 16))
        gs = plt.GridSpec(4, 2, figure=final_fig)
        
        # Plot 1: Loss vs Steps
        ax1 = final_fig.add_subplot(gs[0, :])
        ax1.plot(self.steps, self.losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss vs Steps', fontsize=14)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Loss vs Epochs
        ax2 = final_fig.add_subplot(gs[1, 0])
        ax2.plot(self.epochs, self.losses, 'r-', linewidth=2)
        ax2.set_title('Training Loss vs Epochs', fontsize=14)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Learning Rate vs Steps
        ax3 = final_fig.add_subplot(gs[1, 1])
        ax3.plot(self.steps, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 4: Training vs Validation Loss
        ax4 = final_fig.add_subplot(gs[2, :])
        ax4.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Training')
        
        if any(x is not None for x in self.val_losses):
            # Filter out None values
            val_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
            val_loss_values = [v for v in self.val_losses if v is not None]
            if val_epochs:
                ax4.plot(val_epochs, val_loss_values, 'r-', linewidth=2, label='Validation')
                ax4.legend()
                
        ax4.set_title('Training vs Validation Loss', fontsize=14)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss')
        ax4.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 5: Combined plot with two y-axes
        ax5 = final_fig.add_subplot(gs[3, :])
        color1, color2 = 'blue', 'green'
        
        # Plot loss on left axis
        line1 = ax5.plot(self.epochs, self.losses, color=color1, linewidth=2.5, label='Loss')
        ax5.set_xlabel('Epochs')
        ax5.set_ylabel('Loss', color=color1)
        ax5.tick_params(axis='y', labelcolor=color1)
        
        # Plot learning rate on right axis
        ax6 = ax5.twinx()
        line2 = ax6.plot(self.epochs, self.learning_rates, color=color2, linewidth=2.5, label='Learning Rate')
        ax6.set_ylabel('Learning Rate', color=color2)
        ax6.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper right')
        ax5.set_title('Loss and Learning Rate vs Epochs', fontsize=14)
        ax5.grid(True, linestyle='--', alpha=0.7)
        
        # Add training summary
        if self.epochs:
            epoch_count = max(self.epochs)
            step_count = max(self.steps)
            min_loss = min(self.losses)
            min_loss_epoch = self.epochs[self.losses.index(min_loss)]
            min_loss_step = self.steps[self.losses.index(min_loss)]
            
            # Calculate convergence indicators
            recent_epochs = min(10, len(self.losses))
            recent_losses = self.losses[-recent_epochs:]
            loss_change_pct = ((recent_losses[0] - recent_losses[-1]) / recent_losses[0]) * 100 if recent_losses[0] != 0 else 0
            
            summary_text = (
                f"Training Summary\n"
                f"Total Epochs: {epoch_count:.2f}\n"
                f"Total Steps: {step_count}\n"
                f"Min Loss: {min_loss:.6f} (Epoch {min_loss_epoch:.2f}, Step {min_loss_step})\n"
                f"Recent {recent_epochs} epochs loss change: {loss_change_pct:.2f}%\n"
            )
            
            if len(self.losses) > 20:
                # Add convergence assessment
                last_20_losses = self.losses[-20:]
                std_last_20 = np.std(last_20_losses)
                converged = std_last_20 < 0.01 and loss_change_pct < 1.0
                
                summary_text += f"Convergence status: {'CONVERGED' if converged else 'NOT CONVERGED'}\n"
                if converged:
                    summary_text += f"Loss stabilized with std dev {std_last_20:.6f}"
                else:
                    summary_text += f"Loss still changing significantly (std dev: {std_last_20:.6f})"
            
            plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        final_fig.suptitle('CSM Model Finetuning Metrics', fontsize=16, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        
        # Save the final detailed plot
        final_plot_path = os.path.join(self.output_dir, 'training_metrics_final.png')
        final_fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close(final_fig)
        plt.close(self.fig)
        
        logger.info(f"Final training visualization saved to {final_plot_path}")
        
        return final_plot_path

def load_llama3_tokenizer():
    tokenizer_name = "unsloth/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(bos, tokenizer.bos_token_id), (eos, tokenizer.eos_token_id)],
    )
    return tokenizer

@dataclass
class AudioTextPair:
    audio_path: str
    text: str
    speaker_id: int
    processed_audio: Optional[torch.Tensor] = None
    
    def load_audio(self, sample_rate=24000) -> torch.Tensor:
        if self.processed_audio is not None:
            return self.processed_audio

        waveform, sr = torchaudio.load(self.audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        self.processed_audio = waveform.squeeze(0)
        return self.processed_audio

class CSMDataset(Dataset):
    def __init__(self, data_items, text_tokenizer, audio_tokenizer, device):
        self.data_items = data_items
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.sample_rate = audio_tokenizer.sample_rate
        
    def __len__(self):
        return len(self.data_items)
        
    def tokenize_text_segment(self, text: str, speaker: int):
        text_tokens = self.text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    def tokenize_audio(self, audio: torch.Tensor):
        assert audio.ndim == 1, "Audio must be single channel"
        audio_device = next(self.audio_tokenizer.parameters()).device
        audio = audio.to(audio_device)
        
        with torch.no_grad():
            audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        eos_frame = torch.zeros_like(audio_tokens[:, :1])
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        audio_frame = torch.zeros(audio_tokens.size(1), 33, device=audio_device, dtype=torch.long)
        audio_frame_mask = torch.zeros_like(audio_frame, dtype=torch.bool)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        return audio_frame, audio_frame_mask
    
    def __getitem__(self, idx: int):
        item = self.data_items[idx]
        audio = item.load_audio(self.sample_rate)
        
        text_tokens, text_masks = self.tokenize_text_segment(item.text, item.speaker_id)
        audio_tokens, audio_masks = self.tokenize_audio(audio)
        
        device = audio_tokens.device
        text_tokens = text_tokens.to(device)
        text_masks = text_masks.to(device)
        
        input_tokens = text_tokens
        input_masks = text_masks
        
        target_tokens = torch.cat([text_tokens, audio_tokens], dim=0)
        target_masks = torch.cat([text_masks, audio_masks], dim=0)
        
        if device != self.device:
            input_tokens = input_tokens.to(self.device)
            input_masks = input_masks.to(self.device)
            target_tokens = target_tokens.to(self.device)
            target_masks = target_masks.to(self.device)
        
        return {
            "input_tokens": input_tokens,
            "input_masks": input_masks,
            "target_tokens": target_tokens,
            "target_masks": target_masks,
        }

def collate_fn(batch):
    max_seq_len = MAX_SEQ_LEN
    device = batch[0]["input_tokens"].device
    
    max_input_len = min(max(item["input_tokens"].size(0) for item in batch), max_seq_len)
    max_target_len = min(max(item["target_tokens"].size(0) for item in batch), max_seq_len)

    batch_input_tokens = []
    batch_input_masks = []
    batch_target_tokens = []
    batch_target_masks = []
    
    for item in batch:
        input_tokens = item["input_tokens"][:max_input_len]
        input_masks = item["input_masks"][:max_input_len]
        target_tokens = item["target_tokens"][:max_target_len]
        target_masks = item["target_masks"][:max_target_len]
        if not target_masks[1:, :-1].all(dim=-1).any():
            raise ValueError("A recording has no audio targets after truncation; shorten its text or increase MAX_SEQ_LEN")
        
        input_tokens = F.pad(input_tokens, (0,0,0, max_input_len - input_tokens.size(0)), "constant", 0)
        input_masks = F.pad(input_masks, (0,0,0, max_input_len - input_masks.size(0)), "constant", False)
        
        target_tokens = F.pad(target_tokens, (0,0,0, max_target_len - target_tokens.size(0)), "constant", 0)
        target_masks = F.pad(target_masks, (0,0,0, max_target_len - target_masks.size(0)), "constant", False)
        
        batch_input_tokens.append(input_tokens)
        batch_input_masks.append(input_masks)
        batch_target_tokens.append(target_tokens)
        batch_target_masks.append(target_masks)
    
    return {
        "input_tokens": torch.stack(batch_input_tokens),
        "input_masks": torch.stack(batch_input_masks),
        "target_tokens": torch.stack(batch_target_tokens),
        "target_masks": torch.stack(batch_target_masks),
        "positions": torch.arange(0, max_target_len).unsqueeze(0).repeat(len(batch), 1).to(device)
    }

def transcribe_audio_files():
    from transformers import pipeline
    
    # Cache file path
    cache_file = os.path.join(AUDIO_DIR, "transcription_cache.json")
    
    # Load existing cache
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            logger.info(f"Loaded transcription cache with {len(cache)} entries")
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
            cache = {}
    
    logger.info(f"Transcribing audio files in: {AUDIO_DIR}")
    transcriber = None
    audio_text_pairs = []
    
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav")) \
        + glob.glob(os.path.join(AUDIO_DIR, "*.mp3")) \
        + glob.glob(os.path.join(AUDIO_DIR, "*.flac"))
    audio_files.sort()
    
    if MAX_AUDIO_FILES > 0 and len(audio_files) > MAX_AUDIO_FILES:
        logger.info(f"Found {len(audio_files)} files, limiting to {MAX_AUDIO_FILES}")
        audio_files = audio_files[:MAX_AUDIO_FILES]
    
    cache_hits = 0
    cache_misses = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Create cache key using file path and modification time
            file_stat = os.stat(audio_file)
            cache_key = f"{audio_file}_{file_stat.st_mtime}_{file_stat.st_size}"
            
            # Check if transcription exists in cache
            if cache_key in cache:
                transcription = cache[cache_key]
                cache_hits += 1
                logger.debug(f"Cache hit: {os.path.basename(audio_file)}")
            else:
                # Transcribe the file
                if transcriber is None:
                    transcriber = pipeline("automatic-speech-recognition", model=TRANSCRIPTION_MODEL, device=-1)
                result = transcriber(audio_file, return_timestamps=True, chunk_length_s=30,
                                   stride_length_s=[6, 0], batch_size=32,
                                   generate_kwargs={"language": "<|en|>", "task": "transcribe"})
                transcription = result["text"].strip()
                
                # Save to cache
                cache[cache_key] = transcription
                cache_misses += 1
                logger.info(f"Transcribed: {os.path.basename(audio_file)} -> {transcription}")
            
            audio_text_pairs.append(
                AudioTextPair(audio_path=audio_file, text=transcription, speaker_id=SPEAKER_ID)
            )
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")
    
    # Save updated cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved transcription cache with {len(cache)} entries")
    except Exception as e:
        logger.error(f"Could not save cache file: {e}")
    
    logger.info(f"Processed {len(audio_text_pairs)} audio files (Cache hits: {cache_hits}, Cache misses: {cache_misses})")
    return audio_text_pairs

def prepare_csm_model_for_training():
    logger.info(f"Loading CSM model: {MODEL_NAME}")
    model = Model.from_pretrained(MODEL_NAME).to(DEVICE)

    text_tokenizer = load_llama3_tokenizer()
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=DEVICE)
    mimi.set_num_codebooks(32)
    audio_tokenizer = mimi.eval().requires_grad_(False)
    logger.info("Applying LoRA to model...")
    model = replace_linear_with_lora(
        model,
        r=R,
        alpha=ALPHA,
        dropout=0.01,
        target_linear_names=TARGET_LAYERS
    )
    model.to(DEVICE)
    

    # First, freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False

    # Then, unfreeze only the newly added LoRA parameters.
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True

    return model, text_tokenizer, audio_tokenizer

def finetune(model, dataset):
    if BATCH_SIZE < 1 or GRADIENT_ACCUMULATION_STEPS < 1 or NUM_EPOCHS < 1:
        raise ValueError("Batch size, accumulation steps, and epochs must be positive")
    if MAX_SEQ_LEN > model.backbone.max_seq_len:
        raise ValueError("MAX_SEQ_LEN exceeds the backbone context length")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_epoch, checkpoint_dir = latest_training_checkpoint(OUTPUT_DIR)
    resume_state = None
    if checkpoint_dir is not None:
        metadata = json.loads((checkpoint_dir / "adapter_config.json").read_text())
        if metadata["base_model"] != MODEL_NAME:
            raise ValueError("Checkpoint base model differs from MODEL_NAME")
        logger.info("Resuming from %s (completed epoch %d)", checkpoint_dir, start_epoch)
        load_checkpoint(model, checkpoint_dir)
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(("lora_A", "lora_B")))
        state_path = checkpoint_dir / "training_state.pt"
        if metadata.get("training_state_required"):
            resume_state = torch.load(state_path, map_location="cpu", weights_only=True)
            if resume_state["completed_epochs"] != start_epoch:
                raise ValueError("Checkpoint epoch does not match its training state")
        else:
            logger.warning("Checkpoint has model weights only. Optimizer and scaler history "
                           "are unavailable; continuing with fresh optimizer state.")
    train_dataset, val_dataset = split_dataset(dataset, VALIDATION_FRACTION, SEED)
    split = {"seed": SEED, "train_indices": train_dataset.indices, "validation_indices": val_dataset.indices}
    if hasattr(dataset, "data_items"):
        split["recordings"] = [os.path.abspath(item.audio_path) for item in dataset.data_items]
    split_path = Path(OUTPUT_DIR) / "dataset_split.json"
    if checkpoint_dir is not None:
        saved_split = resume_state["split"] if resume_state else json.loads(split_path.read_text())
        if saved_split.get("recordings") != split.get("recordings"):
            raise ValueError("Recordings differ from the saved training split")
        indices = saved_split["train_indices"] + saved_split["validation_indices"]
        if sorted(indices) != list(range(len(dataset))):
            raise ValueError("Saved training split does not match this dataset")
        split = saved_split
        train_dataset = Subset(dataset, split["train_indices"])
        val_dataset = Subset(dataset, split["validation_indices"])
    with open(os.path.join(OUTPUT_DIR, "dataset_split.json"), "w") as f:
        json.dump(split, f, indent=2)
    logger.info("Training on %d recordings, validating on %d", len(train_dataset), len(val_dataset))
    data_generator = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            generator=data_generator,
                            collate_fn=collate_fn, num_workers=0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    steps_per_epoch = math.ceil(len(dataloader) / GRADIENT_ACCUMULATION_STEPS)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS,
                              num_training_steps=steps_per_epoch * NUM_EPOCHS)
    use_amp = MIXED_PRECISION and torch.device(DEVICE).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    global_step = start_epoch * steps_per_epoch
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state["scaler"] and use_amp:
            scaler.load_state_dict(resume_state["scaler"])
        global_step = resume_state["global_step"]
    elif start_epoch:
        # Weight-only checkpoints have no optimizer history. Continue the LR
        # schedule at the estimated completed update count instead of warming up again.
        scheduler.last_epoch = global_step
        for group, base_lr, schedule in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
            group["lr"] = base_lr * schedule(global_step)
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    visualizer = TrainingVisualizer(OUTPUT_DIR)
    csv_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    history = []
    if checkpoint_dir is not None and os.path.exists(csv_file):
        with open(csv_file, newline="") as f:
            history = [row for row in csv.DictReader(f) if float(row["epoch"]) <= start_epoch]
    with open(csv_file, "w", newline="") as f:
        fields = ["epoch", "global_step", "loss", "learning_rate", "val_loss"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in history)
    for row in history:
        visualizer.epochs.append(float(row["epoch"]))
        visualizer.steps.append(int(row["global_step"]))
        visualizer.losses.append(float(row["loss"]))
        visualizer.learning_rates.append(float(row["learning_rate"]))
        visualizer.val_losses.append(float(row["val_loss"]) if row["val_loss"] else None)
    if USE_WANDB:
        wandb.init(project="csm-finetuning")

    def validate():
        with torch.amp.autocast(device_type=torch.device(DEVICE).type, dtype=torch.float16, enabled=use_amp):
            return calculate_validation_loss(model, val_dataset, DEVICE, BATCH_SIZE,
                                             DECODER_CHUNK_SIZE, collate_fn)

    logger.info("Initial held-out validation loss: %.6f", validate())
    if resume_state is not None:
        torch.set_rng_state(resume_state["torch_rng"])
        if torch.cuda.is_available() and resume_state["cuda_rng"]:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        data_generator.set_state(resume_state["data_rng"])
    if start_epoch >= NUM_EPOCHS:
        logger.info("Checkpoint already completed %d epochs; NUM_EPOCHS=%d. No further epochs needed.",
                    start_epoch, NUM_EPOCHS)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, NUM_EPOCHS):
        running_loss, epoch_frames = 0.0, 0
        window_frames = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
            frames = batch["target_masks"][:, 1:, :model.config.audio_num_codebooks].all(-1).sum().item()
            with torch.amp.autocast(device_type=torch.device(DEVICE).type, dtype=torch.float16, enabled=use_amp):
                loss = forward_and_loss(model, batch, DEVICE, DECODER_CHUNK_SIZE)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch + 1}, batch {step}")
            # Sum frame losses, then normalize accumulated gradients by the actual
            # number of frames, including the final partial accumulation window.
            scaler.scale(loss * frames).backward()
            window_frames += frames
            running_loss += loss.item() * frames
            epoch_frames += frames
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step + 1 == len(dataloader):
                scaler.unscale_(optimizer)
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.div_(window_frames)
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
                window_frames = 0
        val_loss = validate()
        train_loss = running_loss / epoch_frames
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d: train %.6f, held-out validation %.6f", epoch + 1, train_loss, val_loss)
        with open(csv_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, global_step, train_loss, lr, val_loss])
        visualizer.update(epoch + 1, global_step, train_loss, lr, val_loss)
        if USE_WANDB:
            wandb.log({"loss": train_loss, "val_loss": val_loss, "learning_rate": lr, "epoch": epoch + 1})
        save_checkpoint(model, os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch + 1}"), MODEL_NAME,
                        training_state={
                            "completed_epochs": epoch + 1, "global_step": global_step,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "split": split,
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            "data_rng": data_generator.get_state(),
                        })

    model = remove_lora_modules(merge_lora_weights(model))
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(OUTPUT_DIR, "model.safetensors"))
    visualizer.finalize()
    if USE_WANDB:
        wandb.finish()
    return model

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(True)
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    
    audio_text_pairs = transcribe_audio_files()
    if not audio_text_pairs:
        logger.error(f"No audio files found or transcribed in {AUDIO_DIR}")
        return
    if len(audio_text_pairs) < 2:
        raise ValueError("At least two recordings are required for a held-out validation split")
    model, text_tokenizer, audio_tokenizer = prepare_csm_model_for_training()
    
    dataset = CSMDataset(
        audio_text_pairs,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        device=DEVICE
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    finetune(model, dataset)
    logger.info("Finetuning completed successfully!")

if __name__ == "__main__":
    main()
