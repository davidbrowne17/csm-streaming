# CSM - Optimized Streaming/Finetuning Edition

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

Our fork adds **streaming audio generation**, **real-time playback**, and **performance optimizations** to the original implementation.

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* For real-time audio playback: `pip install sounddevice`
* Access to the following Hugging Face models:
  * [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:davidbrowne17/csm-streaming.git
cd csm-streaming
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sounddevice  # Optional: for real-time audio playback

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Llama-3.2-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

Generate a sentence with streaming (chunks are processed and output as they're generated):

```python
import time
from huggingface_hub import hf_hub_download
from generator import Generator, Segment, load_csm_1b, generate_streaming_audio
import torchaudio

# Load the model
generator = load_csm_1b("cuda")

# Generate audio with streaming and real-time playback
generate_streaming_audio(
    generator=generator,
    text="Hello, this is streaming audio generation in action!",
    speaker=0,
    context=[],  # No context needed for basic generation
    output_file="streaming_audio.wav",
    play_audio=True  # Enable real-time playback
)
```

## Usage

Our optimized version offers several ways to use CSM with streaming capabilities:

### 1. Basic Streaming Generation

Generate audio with streaming and save to a file:

```python
from generator import load_csm_1b, generate_streaming_audio

generator = load_csm_1b("cuda")

# Generate with streaming (writes to file as it generates)
generate_streaming_audio(
    generator=generator,
    text="This audio will be generated in chunks for faster response times.",
    speaker=0,
    context=[],
    output_file="streaming_output.wav"
)
```

### 2. Real-time Audio Playback

Generate and play audio in real-time as it's being generated:

```python
from generator import load_csm_1b, generate_streaming_audio

generator = load_csm_1b("cuda")

# Generate with streaming and play in real-time
generate_streaming_audio(
    generator=generator,
    text="You'll hear me speaking as I'm being generated!",
    speaker=0,
    context=[],
    output_file="streaming_output.wav",
    play_audio=True  # Enable real-time playback
)
```

### 3. Low-level Streaming API

For more control, use the low-level streaming API:

```python
from generator import load_csm_1b, Segment
import torchaudio

generator = load_csm_1b("cuda")

# Process audio chunks as they're generated
for audio_chunk in generator.generate_stream(
    text="This is generated chunk by chunk.",
    speaker=0,
    context=[]
):
    # Do something with each chunk as it's generated
    print(f"Received chunk of size: {audio_chunk.shape}")
    
    # You could process or play each chunk here
    # For example, write to a file incrementally
    # Or send over a network connection
```

### 4. Generate with Context

For best results, provide reference audio context:

```python
from generator import load_csm_1b, Segment, generate_streaming_audio
import torchaudio

generator = load_csm_1b("cuda")

# Load reference audio
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create context segments
segments = [
    Segment(
        text="I knew I could trust you.",
        speaker=0,
        audio=load_audio("reference.wav")
    )
]

# Generate with streaming using the context
generate_streaming_audio(
    generator=generator,
    text="Me too, this is some cool stuff huh?",
    speaker=0,
    context=segments,
    output_file="contextual_streaming.wav",
    play_audio=True
)
```

### 5. Regular Generation with Internal Streaming

Use the original API with streaming enabled internally:

```python
from generator import load_csm_1b, Segment
import torchaudio

generator = load_csm_1b("cuda")

# Regular generation but with internal streaming optimization
audio = generator.generate(
    text="This uses internal streaming for faster processing.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
    stream=True  # Enable internal streaming optimization
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### 6. Finetuning
To finetune CSM all you need are some wav audio files with the speaker voice you want to train, just the raw wavs. Place them in a folder called audio_data and run lora.py.
You can configure the exact training params such as batch size, number of epochs and learning rate by modifying the values at the top of lora.py.
You will need a CUDA gpu with at least 12gb of vram depending on your dataset size and training params.

## Performance Optimizations

Our optimized version includes several performance enhancements:

- **Streaming Generation**: Processes and outputs audio in chunks instead of waiting for the entire generation
- **Frame Batching**: Processes multiple frames at once for better GPU utilization
- **Lower Temperature Sampling**: Faster sampling with minimal quality loss
- **Half-precision Inference**: Uses bfloat16/float16 for faster processing
- **CUDA Optimizations**: Enables cuDNN benchmarking and Flash Attention where available
- **Memory Management**: Clears GPU cache before generation to reduce memory pressure

## FAQ

**How much faster is the streaming version?**

The perceived response time is significantly faster since you get the first audio chunks in milliseconds instead of waiting for the entire generation to complete. The actual total generation time is also improved by 20-40% depending on your hardware.

**Does this model come with any voices?**

The model is a base generation model capable of producing a variety of voices but hasn't been fine-tuned on any specific voice. Provide reference audio for best results.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Original Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.

## Streaming and Finetuning Implementation
David Browne
