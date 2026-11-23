#!/usr/bin/env python3
"""F5-TTS 语音克隆合成（Colab 上跑）。

用法：
  python synthesize.py --ref-audio-url <wav 链接> --ref-text-url <txt 链接>
      --gen-text "要说的话" | --gen-text-file /path/to/text.txt | --gen-text-url <txt 链接>
      --out /content/out.wav [--model F5TTS_v1_Base] [--speed 1.0]

要点：
- 参考音用短的（约 9 秒）；F5-TTS 会把超过 12 秒的参考音截断，参考文本须与实际音频对应。
- 生成文本按句子切块逐块生成、块间留 0.3 秒停顿再拼接——长文一次生成会被模型压成快放，
  切块才是正常语速的正确做法；不要靠 --speed 拉伸补救（拉伸快放只是慢速快放）。
- 先走 Python API，失败退 f5-tts CLI。
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import urllib.request
import wave


def download(url, path):
    print("下载 %s -> %s" % (url, path))
    urllib.request.urlretrieve(url, path)
    print("  %d 字节" % os.path.getsize(path))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def ensure_f5():
    if importlib.util.find_spec("f5_tts") is None:
        print("f5-tts 未装，pip 安装中（约 2-4 分钟）...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "f5-tts"],
                       check=True)
        print("f5-tts 安装完成")
    else:
        print("f5-tts 已装")


def ref_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def split_gen_chunks(text, max_len=60):
    """按句末标点切块；过长的句再按逗号/顿号/冒号切；每块不超过 max_len 字。"""
    parts = re.split(r"(?<=[。！？；])", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_len:
            chunks.append(p)
            continue
        sub = re.split(r"(?<=[，、：])", p)
        buf = ""
        for s in sub:
            if len(buf) + len(s) <= max_len:
                buf += s
            else:
                if buf:
                    chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    return chunks


def concat_wavs(wav_files, out_path, pause_sec=0.3):
    """把多段 wav 按顺序拼起来，段间插 pause_sec 秒静音。"""
    frames = b""
    params = None
    for i, f in enumerate(wav_files):
        with wave.open(f, "rb") as w:
            if params is None:
                params = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            frames += w.readframes(w.getnframes())
        if i < len(wav_files) - 1:
            nch, sw, sr = params
            frames += b"\x00" * int(sr * pause_sec) * sw * nch
    nch, sw, sr = params
    with wave.open(out_path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames)
    return out_path


def infer_api_chunks(ref_audio, ref_text, chunks, work_dir, model, speed=1.0):
    from f5_tts.api import F5TTS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备：%s，模型：%s，语速：%s，共 %d 块" % (device, model, speed, len(chunks)))
    f5 = F5TTS(model=model, device=device)
    outs = []
    for i, c in enumerate(chunks):
        out_path = os.path.join(work_dir, "chunk_%02d.wav" % i)
        print("第 %d/%d 块（%d 字）：%s" % (i + 1, len(chunks), len(c), c[:30]))
        f5.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=c,
                 file_wave=out_path, speed=speed)
        outs.append(out_path)
    return outs


def infer_cli_chunks(ref_audio, ref_text, chunks, work_dir, model, speed=1.0):
    outs = []
    for i, c in enumerate(chunks):
        chunk_dir = os.path.join(work_dir, "cli_%02d" % i)
        os.makedirs(chunk_dir, exist_ok=True)
        print("第 %d/%d 块（CLI，%d 字）：%s" % (i + 1, len(chunks), len(c), c[:30]))
        cmd = [sys.executable, "-m", "f5_tts.infer.infer_cli",
               "--model", model,
               "--ref_audio", ref_audio,
               "--ref_text", ref_text,
               "--gen_text", c,
               "--speed", str(speed),
               "--output_dir", chunk_dir]
        subprocess.run(cmd, check=True)
        wavs = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".wav"))
        if not wavs:
            raise RuntimeError("CLI 第 %d 块未产出 wav" % (i + 1))
        src = os.path.join(chunk_dir, wavs[-1])
        dst = os.path.join(work_dir, "chunk_%02d.wav" % i)
        os.replace(src, dst)
        outs.append(dst)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio-url", required=True)
    ap.add_argument("--ref-text-url", required=True)
    ap.add_argument("--gen-text", default="")
    ap.add_argument("--gen-text-file", default="")
    ap.add_argument("--gen-text-url", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="语速：小于 1 变慢，大于 1 变快；正常语速靠切块，不靠它")
    args = ap.parse_args()

    ref_audio = "/content/ref.wav"
    ref_text_file = "/content/ref.txt"
    download(args.ref_audio_url, ref_audio)
    download(args.ref_text_url, ref_text_file)
    ref_text = read_text(ref_text_file)
    print("参考文本 %d 字：%s" % (len(ref_text), ref_text[:40]))
    d = ref_duration(ref_audio)
    print("参考音频 %.1f 秒" % d)
    if d > 12:
        print("注意：参考音超过 12 秒，F5-TTS 会截断，只用前 12 秒，"
              "参考文本须与前 12 秒的实际内容对应，否则音色会歪")

    if args.gen_text_url:
        download(args.gen_text_url, "/content/gen.txt")
        gen_text = read_text("/content/gen.txt")
    elif args.gen_text_file:
        gen_text = read_text(args.gen_text_file)
    else:
        gen_text = args.gen_text
    if not gen_text:
        print("没有生成文本")
        sys.exit(2)
    print("生成文本 %d 字" % len(gen_text))

    chunks = split_gen_chunks(gen_text)
    print("切成 %d 块，块长：%s" % (len(chunks), [len(c) for c in chunks]))

    ensure_f5()
    work_dir = "/content/chunks"
    os.makedirs(work_dir, exist_ok=True)
    try:
        outs = infer_api_chunks(ref_audio, ref_text, chunks, work_dir,
                                args.model, speed=args.speed)
    except Exception as e:
        print("API 推理出错（%s），退 CLI" % e)
        outs = infer_cli_chunks(ref_audio, ref_text, chunks, work_dir,
                                args.model, speed=args.speed)

    concat_wavs(outs, args.out)
    with wave.open(args.out, "rb") as w:
        dur = w.getnframes() / w.getframerate()
    print("DONE %s（%.1f 秒，%d 字节）" % (args.out, dur, os.path.getsize(args.out)))


if __name__ == "__main__":
    main()
