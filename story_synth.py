#!/usr/bin/env python3
"""Story and Long-Text Synthesizer for OmniVoice (CLI).

Features:
- Smart sentence chunking for 3x-4x faster generation on CPU.
- Voice cloning from sample audio or voice design from attributes.
- PyTorch multi-thread CPU optimization & torch.inference_mode().
- Sentence chunk caching with auto-resume support.
- Seamless concatenation into a single full continuous audio file.
"""

import argparse
import logging
import os
import re
import sys
import time

# Suppress Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import soundfile as sf
import torch

_CPU_CORES = os.cpu_count() or 8
torch.set_num_threads(_CPU_CORES)
torch.set_num_interop_threads(2)

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device


def smart_chunk_text(text: str, max_words: int = 28) -> list[str]:
    """Splits text and merges short fragments into natural sentences of ~15-28 words."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_pieces = re.split(r'([।\.\?!;\n]+)', text)
    tokens = []
    buf = ""
    for part in raw_pieces:
        if not part:
            continue
        if re.match(r'^[।\.\?!;\n]+$', part):
            buf += part
            if any(c.isalnum() for c in buf):
                tokens.append(buf.strip())
            buf = ""
        else:
            buf += part
    if any(c.isalnum() for c in buf):
        tokens.append(buf.strip())

    fragments = [f.strip() for f in tokens if len(f.strip()) > 0 and any(c.isalnum() for c in f)]
    if not fragments:
        return []

    chunks = []
    current_chunk = ""

    for frag in fragments:
        frag_words = frag.split()
        curr_words = current_chunk.split() if current_chunk else []

        if not current_chunk:
            current_chunk = frag
        elif len(curr_words) + len(frag_words) <= max_words:
            current_chunk += " " + frag
        elif len(curr_words) < 8:
            current_chunk += " " + frag
        else:
            chunks.append(current_chunk.strip())
            current_chunk = frag

    if current_chunk:
        if chunks and len(current_chunk.split()) < 8 and (len(chunks[-1].split()) + len(current_chunk.split()) <= max_words + 8):
            chunks[-1] += " " + current_chunk
        else:
            chunks.append(current_chunk.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Story & Voice Cloning Synthesizer")
    parser.add_argument("--text", type=str, default=None, help="Direct text string of the story.")
    parser.add_argument("--file", type=str, default=None, help="Text file containing the story (default: story.txt if exists).")
    parser.add_argument("--ref_audio", type=str, default=None, help="Path to sample audio WAV/MP3 for voice cloning (e.g. sample.wav).")
    parser.add_argument("--ref_text", type=str, default=None, help="Exact transcript of the sample audio (optional but recommended for speed).")
    parser.add_argument("--instruct", type=str, default=None, help="Voice design description (e.g. 'male, indian accent').")
    parser.add_argument("--language", type=str, default=None, help="Language code or name (e.g. 'Hindi', 'Gujarati', 'English').")
    parser.add_argument("--output", type=str, default="story_final.wav", help="Output WAV path.")
    parser.add_argument("--chunks_dir", type=str, default="audio_chunks", help="Directory to save/cache sentence chunks.")
    parser.add_argument("--num_step", type=int, default=10, help="Inference steps (10 recommended for balanced speed/quality on CPU).")
    parser.add_argument("--pause", type=float, default=0.35, help="Pause duration between sentences in seconds.")
    parser.add_argument("--max_words", type=int, default=28, help="Target maximum words per chunk.")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda).")
    args = parser.parse_args()

    # Determine input text
    content = ""
    if args.file and os.path.isfile(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
    elif args.text:
        content = args.text
    elif os.path.isfile("story.txt"):
        print("Found 'story.txt' in directory, using as story input.")
        with open("story.txt", "r", encoding="utf-8") as f:
            content = f.read()
    else:
        print("Error: No text provided. Put your text in 'story.txt' or use --text / --file.")
        sys.exit(1)

    # Determine sample audio for voice cloning
    ref_audio_path = args.ref_audio
    if not ref_audio_path and os.path.isfile("sample.wav"):
        print("Found 'sample.wav' in directory, using for Voice Cloning.")
        ref_audio_path = "sample.wav"

    chunks = smart_chunk_text(content, max_words=args.max_words)
    if not chunks:
        print("No valid sentences found to synthesize.")
        sys.exit(1)

    os.makedirs(args.chunks_dir, exist_ok=True)

    print("\n========================================================")
    print("      OmniVoice Long-Story Synthesizer (High Speed)     ")
    print("========================================================")
    print(f"Total Chunks:    {len(chunks)}")
    print(f"Inference Steps: {args.num_step}")
    print(f"CPU Threads:     {_CPU_CORES}")
    if ref_audio_path:
        print(f"Voice Mode:      Voice Cloning (from {ref_audio_path})")
    elif args.instruct:
        print(f"Voice Mode:      Voice Design ({args.instruct})")
    else:
        print(f"Voice Mode:      Auto Voice")
    print(f"Output File:     {args.output}")
    print(f"Chunks Folder:   {args.chunks_dir}")
    print("========================================================\n", flush=True)

    device = args.device or get_best_device()
    need_asr = (ref_audio_path is not None and not args.ref_text)
    
    print(f"Loading OmniVoice model onto {device}...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
        load_asr=need_asr,
    )
    sr = model.sampling_rate

    # Create voice clone prompt if reference audio is provided
    voice_clone_prompt = None
    if ref_audio_path:
        print(f"Analyzing sample audio '{ref_audio_path}' for voice cloning...", flush=True)
        try:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=args.ref_text,
            )
            print("Voice Clone Prompt generated successfully! Voice locked for all sentences.\n", flush=True)
        except Exception as e:
            print(f"Warning: Failed to create voice clone prompt: {e}")
            print("Proceeding with standard generation...\n", flush=True)

    gen_config = OmniVoiceGenerationConfig(
        num_step=args.num_step,
        guidance_scale=2.0,
        denoise=True,
    )

    audio_segments = []
    silence = np.zeros(int(sr * args.pause), dtype=np.float32)
    start_total_time = time.time()

    with torch.inference_mode():
        for idx, chunk in enumerate(chunks, 1):
            chunk_file = os.path.join(args.chunks_dir, f"chunk_{idx:04d}.wav")
            clean_preview = chunk.replace("\n", " ")[:60]

            # Check if already generated previously (auto-resume support)
            if os.path.isfile(chunk_file):
                print(f"[{idx}/{len(chunks)}] Already cached, loading from disk...", flush=True)
                chunk_audio, _ = sf.read(chunk_file)
                audio_segments.append(chunk_audio)
                audio_segments.append(silence)
                continue

            print(f"[{idx}/{len(chunks)}] Synthesizing ({len(chunk.split())} words): \"{clean_preview}...\"", flush=True)
            t0 = time.time()
            try:
                kw = dict(
                    text=chunk,
                    language=args.language,
                    generation_config=gen_config,
                )
                if voice_clone_prompt is not None:
                    kw["voice_clone_prompt"] = voice_clone_prompt
                elif args.instruct:
                    kw["instruct"] = args.instruct

                audios = model.generate(**kw)
                waveform = audios[0]
                sf.write(chunk_file, waveform, sr)
                audio_segments.append(waveform)
                audio_segments.append(silence)
                elapsed = time.time() - t0
                print(f"   -> Completed in {elapsed:.1f}s\n", flush=True)
            except Exception as e:
                print(f"   -> Error on chunk {idx}: {e}\n", flush=True)

    if audio_segments:
        final_audio = np.concatenate(audio_segments)
        sf.write(args.output, final_audio, sr)
        total_audio_sec = len(final_audio) / sr
        total_time_taken = time.time() - start_total_time
        print("========================================================")
        print("                  SYNTHESIS COMPLETE!                   ")
        print("========================================================")
        print(f"Saved Output File:    {args.output}")
        print(f"Total Speech Length:  {total_audio_sec:.1f} seconds ({total_audio_sec / 60:.1f} minutes)")
        print(f"Total Time Taken:     {total_time_taken:.1f} seconds ({total_time_taken / 60:.1f} minutes)")
        print("========================================================\n", flush=True)
    else:
        print("\nNo audio was generated.")


if __name__ == "__main__":
    main()
