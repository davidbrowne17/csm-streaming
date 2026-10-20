import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from finetuning import load_checkpoint, merge_lora_weights, remove_lora_modules
from models import Model


def find_latest_checkpoint(dir_path):
    checkpoints = [(int(match.group(1)), path) for path in Path(dir_path).iterdir()
                   if path.is_dir() and (match := re.fullmatch(r'checkpoint-epoch-(\d+)', path.name))]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {dir_path}')
    return max(checkpoints)[1]


def load_checkpoint_and_merge(checkpoint_path=None, output_dir='finetuned_model', device=None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else find_latest_checkpoint(output_dir)
    config_path = checkpoint_path / 'adapter_config.json'
    if not config_path.exists():
        raise ValueError('Checkpoint has no adapter_config.json. Legacy bridge checkpoints cannot be '
                         'safely merged automatically; retrain with the corrected decoder objective.')
    config = json.loads(config_path.read_text())
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model.from_pretrained(config['base_model']).to(device)
    load_checkpoint(model, checkpoint_path)
    model = remove_lora_modules(merge_lora_weights(model))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()},
              str(output / 'model.safetensors'))
    print(f'Merged model saved to {output / "model.safetensors"}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Defaults to the latest checkpoint in --output-dir')
    parser.add_argument('--output-dir', default='finetuned_model')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    load_checkpoint_and_merge(args.checkpoint, args.output_dir, args.device)
