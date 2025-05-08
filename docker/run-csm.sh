#!/bin/bash
# Syntax: run-csm.sh [/tmp/test-docker-setup.py]

CSM_MODEL_PATH=$(realpath ~/.cache/huggingface/hub/models--sesame--csm-1b/snapshots/*/model.safetensors)
CSM_CONFIG_PATH=$(realpath ~/.cache/huggingface/hub/models--sesame--csm-1b/snapshots/*/config.json)

## if using pulseaudio instead of pipewire, try:
#-v $XDG_RUNTIME_DIR/pulse/native:/tmp/pulse-native
#-e PULSE_SERVER=unix:/tmp/pulse-native
## or remote socket? (requires load-module module-native-protocol-tcp etc)
#-e PULSE_SERVER=tcp:host.docker.internal

docker run --rm -it -p 8000:8000 --name csm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --device /dev/snd \
  --group-add audio \
  -v $XDG_RUNTIME_DIR/pipewire-0:/tmp/pipewire \
  -e PIPEWIRE_REMOTE=unix:/tmp/pipewire \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/pip:/root/.cache/pip \
  -v $CSM_MODEL_PATH:/csm/finetuned_model/model.safetensors \
  -v $CSM_CONFIG_PATH:/csm/finetuned_model/config.json \
  -v ./models:/csm/models \
  -v ./config:/csm/config \
  -v ./companion.db:/csm/companion.db \
  csm-streaming "$@"

