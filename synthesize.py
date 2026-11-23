#!/usr/bin/env python3
"""小然音克隆：在 Colab T4 上用 F5-TTS 把参考音（小然音）克隆到目标文本。

用法（本地触发）：
  colab run --gpu T4 --keep -s xiaoran-trial-01 synthesize.py \
      --ref-audio-url https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/main/ref/ref_long.wav \
      --ref-text-url  https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/main/ref/ref_long.txt \
      --gen-text "官方说鱼可以留下来、只是要往后退一格，然然听着心里踏实多啦，尾巴轻轻摇着呢！" \
      --out /content/xiaoran_trial.wav

流程：下载参考音频与参考文本 -> 确保 f5-tts 已装（没有就 pip 装）->
F5-TTS 推理（先 Python API，不行再退 CLI）-> 写出 wav，打印 DONE 与时长。
参考文本必须与参考音频逐字一致（音频就是照这段文字念的），否则克隆会走样。
"""
import argparse
import os
import subprocess
import sys
import urllib.request


def download(url, dest):
    print("下载 %s -> %s" % (url, dest))
    urllib.request.urlretrieve(url, dest)
    print("  %d 字节" % os.path.getsize(dest))


def ensure_f5_tts():
    try:
        import f5_tts  # noqa: F401
        print("f5-tts 已装")
        return
    except ImportError:
        print("f5-tts 未装，pip 安装中（约 2-4 分钟）...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "f5-tts"],
            capture_output=True, text=True, timeout=1200)
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        if r.returncode != 0:
            raise SystemExit("pip 安装 f5-tts 失败，退出码 %d" % r.returncode)
        import f5_tts  # noqa: F401
        print("f5-tts 安装完成")


def infer_api(ref_audio, ref_text, gen_text, out_path, model, speed=1.0):
    from f5_tts.api import F5TTS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备：%s，模型：%s，语速：%s" % (device, model, speed))
    f5 = F5TTS(model=model, device=device)
    wav, sr, _spec = f5.infer(
        ref_file=ref_audio, ref_text=ref_text, gen_text=gen_text,
        file_wave=out_path, speed=speed)
    print("API 推理完成：%s（%d Hz，%d 采样）" % (out_path, sr, len(wav)))
    return out_path


def infer_cli(ref_audio, ref_text, gen_text, out_path, model, speed=1.0):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    out_name = os.path.basename(out_path)
    cmd = [sys.executable, "-m", "f5_tts.infer.infer_cli",
           "--model", model,
           "--ref_audio", ref_audio,
           "--ref_text", ref_text,
           "--gen_text", gen_text,
           "--speed", str(speed),
           "--output_dir", out_dir]
    print("API 不可用，退 CLI：%s" % " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("CLI 推理失败，退出码 %d" % r.returncode)
    # CLI 的输出文件名不固定，找目录里最新的 wav
    cands = sorted(
        (os.path.join(out_dir, f) for f in os.listdir(out_dir)
         if f.lower().endswith(".wav")),
        key=os.path.getmtime)
    if not cands:
        raise SystemExit("CLI 跑完但没找到输出 wav")
    newest = cands[-1]
    if os.path.abspath(newest) != os.path.abspath(out_path):
        os.replace(newest, out_path)
    print("CLI 推理完成：%s" % out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="F5-TTS 克隆小然音")
    ap.add_argument("--ref-audio-url", required=True)
    ap.add_argument("--ref-text-url", required=True)
    ap.add_argument("--gen-text", required=True, help="要合成的目标文本")
    ap.add_argument("--out", default="/content/xiaoran_trial.wav")
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="语速：小于 1 变慢（如 0.8），大于 1 变快")
    args = ap.parse_args()

    workdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(workdir, exist_ok=True)
    ref_audio = os.path.join(workdir, "ref.wav")
    ref_text_path = os.path.join(workdir, "ref.txt")
    download(args.ref_audio_url, ref_audio)
    download(args.ref_text_url, ref_text_path)
    ref_text = open(ref_text_path, encoding="utf-8").read().strip()
    print("参考文本 %d 字：%s..." % (len(ref_text), ref_text[:40]))

    ensure_f5_tts()
    try:
        infer_api(ref_audio, ref_text, args.gen_text, args.out, args.model,
                  speed=args.speed)
    except Exception as e:
        print("API 推理出错（%s），退 CLI" % e)
        infer_cli(ref_audio, ref_text, args.gen_text, args.out, args.model,
                  speed=args.speed)

    import wave
    with wave.open(args.out, "rb") as w:
        secs = w.getnframes() / w.getframerate()
    print("DONE %s（%.1f 秒，%d 字节）" % (args.out, secs, os.path.getsize(args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
