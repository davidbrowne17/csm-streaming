import json
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Subset
from safetensors.torch import load_file, save_file

TARGET_LAYERS = ('q_proj', 'k_proj', 'v_proj', 'output_proj', 'w1', 'w2', 'w3')


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=32, alpha=32, dropout=0.0, bias=False):
        super().__init__()
        if r <= 0 or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        self.in_features, self.out_features = in_features, out_features
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B)


def replace_linear_with_lora(model, r=32, alpha=32, dropout=0.0, target_linear_names=TARGET_LAYERS):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.split('.')[-1] in target_linear_names:
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            layer = LoRALinear(module.in_features, module.out_features, r, alpha, dropout,
                               module.bias is not None).to(device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(module.weight)
                if module.bias is not None:
                    layer.bias.copy_(module.bias)
            setattr(parent, child_name, layer)
            replaced.append(name)
    if not replaced:
        raise ValueError('No matching LoRA layers found')
    return model


def merge_lora_weights(model):
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, LoRALinear):
                layer.weight.add_((layer.lora_B.float() @ layer.lora_A.float()).to(layer.weight.dtype),
                                  alpha=layer.scaling)
                layer.lora_B.zero_()
    return model


def remove_lora_modules(module):
    if isinstance(module, LoRALinear):
        layer = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None).to(
            device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(module.weight)
            if module.bias is not None:
                layer.bias.copy_(module.bias)
        return layer.train(module.training)
    for name, child in list(module.named_children()):
        setattr(module, name, remove_lora_modules(child))
    return module


def latest_training_checkpoint(directory):
    candidates = []
    for path in Path(directory).glob('checkpoint-epoch-*'):
        match = re.fullmatch(r'checkpoint-epoch-(\d+)', path.name)
        if not match or not path.is_dir():
            continue
        if not (path / 'model.safetensors').is_file() or not (path / 'adapter_config.json').is_file():
            continue
        config = json.loads((path / 'adapter_config.json').read_text())
        if config.get('training_state_required') and not (path / 'training_state.pt').is_file():
            continue
        candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))


def save_checkpoint(model, directory, base_model, training_state=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layers = {name: {'r': layer.r, 'alpha': layer.alpha, 'dropout': layer.dropout.p}
              for name, layer in model.named_modules() if isinstance(layer, LoRALinear)}
    if not layers:
        raise ValueError('No adapters to save')
    # Metadata is the completion marker; publish it only after every payload.
    metadata_path = directory / 'adapter_config.json'
    metadata_path.unlink(missing_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()},
              str(directory / 'model.safetensors'))
    if training_state is not None:
        torch.save(training_state, directory / 'training_state.pt')
    temporary = directory / 'adapter_config.json.tmp'
    temporary.write_text(json.dumps(
        {'format_version': 1, 'base_model': base_model, 'layers': layers,
         'training_state_required': training_state is not None}, indent=2) + '\n')
    temporary.replace(metadata_path)


def load_checkpoint(model, directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'adapter_config.json').read_text())
    if metadata['format_version'] != 1:
        raise ValueError('Unsupported adapter checkpoint format')
    for name, config in metadata['layers'].items():
        layer = model.get_submodule(name)
        if not isinstance(layer, (nn.Linear, LoRALinear)):
            raise ValueError(f'Expected a base Linear layer at {name}')
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(layer.in_features, layer.out_features,
                bias=layer.bias is not None, **config).to(device=layer.weight.device, dtype=layer.weight.dtype))
    # Full checkpoints must match exactly; never discard trained tensors silently.
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    return model


def split_dataset(dataset, validation_fraction=0.1, seed=42):
    if len(dataset) < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Provide at least two recordings and a validation fraction between 0 and 1')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    count = min(len(dataset) - 1, max(1, round(len(dataset) * validation_fraction)))
    return Subset(dataset, indices[count:]), Subset(dataset, indices[:count])


def forward_and_loss(model, batch, device, decoder_chunk_size=32):
    """Next-frame backbone CE + causal within-frame decoder CE, including EOS=0.

    Decoder position 0 contains the preceding backbone state. Positions 1..K-1
    contain ground truth codebooks 0..K-2 and predict codebooks 1..K-1.
    No inference KV caches may be enabled during training.
    """
    if decoder_chunk_size < 1:
        raise ValueError('decoder_chunk_size must be positive')
    for transformer in (model.backbone, model.decoder):
        if transformer.caches_are_enabled():
            raise ValueError('Disable inference KV caches before training')
    tokens = batch['target_tokens'].to(device)
    masks = batch['target_masks'].to(device)
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    input_masks, label_masks = masks[:, :-1], masks[:, 1:]
    k = model.config.audio_num_codebooks
    active = label_masks[..., :k].all(dim=-1)
    if inputs.size(1) == 0 or not active.any():
        raise ValueError('Batch has no audio targets after truncation')
    if (label_masks[..., :k].any(dim=-1) != active).any():
        raise ValueError('Audio targets must contain every codebook')

    h = (model._embed_tokens(inputs) * input_masks.unsqueeze(-1)).sum(dim=2)
    seq_len = inputs.size(1)
    positions = torch.arange(seq_len, device=device)
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    # Right padding cannot influence earlier real tokens under the causal mask.
    backbone = model.backbone(h, input_pos=positions, mask=causal.unsqueeze(0))
    frame_h = backbone[active]
    frame_labels = labels[..., :k][active]

    def chunk_loss(states, targets):
        loss = F.cross_entropy(model.codebook0_head(states).float(), targets[:, 0], reduction='sum')
        if k > 1:
            offsets = torch.arange(k - 1, device=targets.device) * model.config.audio_vocab_size
            audio = model.audio_embeddings(targets[:, :-1] + offsets)
            decoder_input = model.projection(torch.cat((states.unsqueeze(1), audio), dim=1))
            mask = torch.ones(k, k, dtype=torch.bool, device=targets.device).tril().unsqueeze(0)
            decoded = model.decoder(decoder_input, input_pos=torch.arange(k, device=targets.device), mask=mask)
            for i in range(1, k):
                logits = decoded[:, i] @ model.audio_head[i - 1]
                loss = loss + F.cross_entropy(logits.float(), targets[:, i], reduction='sum')
        return loss

    total = frame_h.new_zeros((), dtype=torch.float32)
    for start in range(0, frame_h.size(0), decoder_chunk_size):
        states = frame_h[start:start + decoder_chunk_size]
        targets = frame_labels[start:start + decoder_chunk_size]
        # Recompute decoder chunks in backward instead of retaining every frame's
        # transformer activations and vocabulary logits for the entire utterance.
        if torch.is_grad_enabled():
            total = total + checkpoint(chunk_loss, states, targets, use_reentrant=False)
        else:
            total = total + chunk_loss(states, targets)
    return total / (frame_h.size(0) * k)


def calculate_validation_loss(model, dataset, device, batch_size=1, decoder_chunk_size=32, collate_fn=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for batch in loader:
                n = batch['target_masks'][:, 1:, :model.config.audio_num_codebooks].all(-1).sum().item()
                loss = forward_and_loss(model, batch, device, decoder_chunk_size)
                total += loss.item() * n
                count += n
    finally:
        model.train(was_training)
    if not count:
        raise ValueError('Validation set has no audio targets')
    return total / count
