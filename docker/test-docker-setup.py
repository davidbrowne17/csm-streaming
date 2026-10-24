#!/usr/bin/env python
import sounddevice as sd
import torch
import runpy

sndlst = sd.query_devices()
print("== sound devices\n", sndlst)
if len(sndlst) == 0:
    print("did you pass --device=/dev/snd?")

print("== GPUs")
if not torch.cuda.is_available():
    print("cuda not available!")
    exit(1)

cnt = torch.cuda.device_count()

for i in range(cnt):
    print("dev", i, "=", torch.cuda.get_device_name(i))

if cnt == 0:
    print("did you pass --device=/dev/dri?")
    exit(1)
else:
    runpy.run_module("bitsandbytes", run_name="__main__")

