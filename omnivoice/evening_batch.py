#!/usr/bin/env python3
"""OmniVoice 晚间批量配音：每天晚上一次性生成当日整篇日记的音频。

用法（Colab 里按顺序跑）：
  1. !pip install omnivoice
  2. 把本目录上传到 Colab（或从 GitHub 拉取），参考音频与声音 prompt 已备好
  3. python3 evening_batch.py --date 2026-11-28 --diary ~/workspace/dafeiyu/diary/2026-11-28.md

约定：
  - 声音：维里奈（Verina），参考音频 ref/verina_selfintro.wav，
    声音 prompt 存 verina_voice.pt（首次运行自动从参考音频生成并保存，
    之后直接加载，不用重复传参考音频、不用重新转写）。
  - 日记文本：读当日 diary/*.md，清理 markdown 后按句切块生成，
    段间 0.6 秒静音拼接（与现有日记音频口径一致）。
  - 输出：当日总音频 <date>-omnivoice.wav（16-bit mono 24000 Hz WAV，
    与 OmniVoice 输出采样率一致），另转 MP3 供博客 frontmatter 使用。
  - 模型：k2-fsa/OmniVoice，device cuda:0，fp16；首次加载从 HuggingFace
    下载约 3.3GB，之后走本地缓存。
"""
import argparse
import re
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path(__file__).resolve().parent
REF_AUDIO = BASE / "ref" / "verina_selfintro.wav"
VOICE_PROMPT = BASE / "verina_voice.pt"
MODEL_NAME = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
GAP_SECONDS = 0.6
# 每块最多约 400 字，和 Mimo 长文切块口径一致，避免单次生成过长
MAX_CHUNK_CHARS = 400


def clean_for_speech(text):
    """把日记 markdown 转成适合朗读的纯文本（与 diary_audio.py 同口径）。"""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.、]\s+", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = line.replace("**", "").replace("*", "").replace("`", "")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return " ".join(out)


def split_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """按句号/问号/感叹号切块，每块不超过 max_chars 字。"""
    sentences = re.split(r"(?<=[。！？；])", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) > max_chars and cur:
            chunks.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="日期，如 2026-11-28")
    ap.add_argument("--diary", required=True, help="当日日记 markdown 路径")
    ap.add_argument("--out-dir", default=None, help="输出目录，默认与 diary 同级的 audio 目录")
    ap.add_argument("--text", default=None, help="直接指定要配音的文字（调试用，优先于 --diary）")
    args = ap.parse_args()

    if args.text:
        full_text = args.text.strip()
    else:
        diary = Path(args.diary)
        full_text = clean_for_speech(diary.read_text(encoding="utf-8"))
    chunks = split_chunks(full_text)
    print(f"全文 {len(full_text)} 字，切成 {len(chunks)} 块")

    from omnivoice import OmniVoice, VoiceClonePrompt

    model = OmniVoice.from_pretrained(
        MODEL_NAME, device_map="cuda:0", dtype=torch.float16, load_asr=True,
    )
    if VOICE_PROMPT.exists():
        prompt = VoiceClonePrompt.load(str(VOICE_PROMPT))
        print(f"已加载声音 prompt：{VOICE_PROMPT}")
    else:
        print("首次运行：从参考音频生成声音 prompt……")
        prompt = model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO))
        prompt.save(str(VOICE_PROMPT))
        print(f"已保存声音 prompt：{VOICE_PROMPT}")

    out_dir = Path(args.out_dir) if args.out_dir else Path.home() / "workspace" / "dafeiyu" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{args.date}-omnivoice.wav"

    gap = np.zeros(int(SAMPLE_RATE * GAP_SECONDS), dtype=np.float32)
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] 生成 {len(chunk)} 字……")
        audio = model.generate(text=chunk, voice_clone_prompt=prompt)
        arr = np.asarray(audio[0], dtype=np.float32)
        pieces.append(arr)
        if i < len(chunks):
            pieces.append(gap)
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    sf.write(str(wav_path), merged, SAMPLE_RATE)
    print(f"已写出：{wav_path}（{len(merged)/SAMPLE_RATE:.1f} 秒）")
    # 顺手转一份 MP3 供博客使用
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
             "-q:a", "2", str(mp3_path)],
            check=True, capture_output=True,
        )
        print(f"已写出：{mp3_path}")
    except Exception as e:
        print(f"MP3 转换跳过（{e}），WAV 已就绪")


if __name__ == "__main__":
    main()
