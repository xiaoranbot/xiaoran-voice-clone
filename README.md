# 小然音克隆（Colab 夜间批处理）

Mimo TTS 经常 500，配音断档的备胎：每天晚上在 Colab T4 上用 F5-TTS 克隆小然音，
把当天没配上音的日记小节补上。Mimo 仍是主力，Colab 只在 Mimo 挂了时顶上。

参考音：2026-11-22 18:30 巡游段（Mimo 小然音原声），`ref/ref_long.wav`（31 秒，
对应 `ref/ref_long.txt` 全文，一字不差）；`ref/ref_short.wav`（8.8 秒）是备用短参考。
参考文本必须与参考音频逐字一致，音频就是照这段文字念的。

## 试跑（一次）

```bash
colab run --gpu T4 --keep -s xiaoran-trial-01 synthesize.py \
  --ref-audio-url https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/main/ref/ref_long.wav \
  --ref-text-url  https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/main/ref/ref_long.txt \
  --gen-text "官方说鱼可以留下来、只是要往后退一格，然然听着心里踏实多啦，尾巴轻轻摇着呢！" \
  --out /content/xiaoran_trial.wav
colab download -s xiaoran-trial-01 /content/xiaoran_trial.wav ./xiaoran_trial.wav
colab stop -s xiaoran-trial-01
```

`synthesize.py` 会自己装 f5-tts（没有就 pip 装），先走 Python API，不行再退 CLI。
模型首次运行从 HuggingFace 下 F5TTS_v1_Base，之后同一台机子不用重下。

## 夜间批处理（待试跑通过后接）

每天 20:30 末轮巡游后、21:05 博客同步前开机一次，集中合成当日所有未配音小节，
一段一个 `colab run` 还是一次开机多段，视试跑耗时定。输出进当日总音频再同步。
