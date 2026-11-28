# OmniVoice 晚间批量配音（ran-blog 日记）

2026-11-28 主人定：日记配音改用这套 OmniVoice 配置，每天晚上一次性批量生成，不再每轮巡游增量生成（避免多次开机浪费）。

## 配置（已备份，直接用）

- 引擎：OmniVoice（k2-fsa/OmniVoice，Apache 2.0 代码；预训练模型 CC-BY-NC，仅限非商业用途）
- 声音：维里奈（Verina，鸣潮），参考音频 `ref/verina_selfintro.wav`（20 秒自我介绍）+ 转写 `ref/verina_selfintro.txt`
- 声音 prompt：`verina_voice.pt`（首次运行由参考音频生成并保存，之后直接加载，不用重复传音频、不用重新转写）
- 模型：k2-fsa/OmniVoice，cuda:0，fp16；首次加载从 HuggingFace 下载约 3.3GB，之后走本地缓存
- 分块：全文按句切，每块最多约 400 字；段间 0.6 秒静音拼接（与现有日记音频口径一致）
- 输出：`<date>-omnivoice.wav`（16-bit mono 24000 Hz）+ 同名 `.mp3`（供博客 frontmatter audio 字段）

## 每晚流程

1. 打开 Colab 笔记本 `OmniVoice-Dub.ipynb`（见本目录），连接 GPU（T4 即可）
2. 按顺序运行：装包 → 传本目录（或从 GitHub 拉）→ 跑 `evening_batch.py --date <当日> --diary <当日日记 md>`
3. 生成的 `<date>-omnivoice.wav` / `.mp3` 拷回 `~/workspace/dafeiyu/audio/`，博客同步时随 frontmatter 一起发

## 注意

- Gradio Demo 的 `--share` 在 Colab 里隧道连不上（2026-11-28 实测 TLS 失败），不要走 Demo 网页，改走本目录的 Python API 批量脚本。
- Colab 提示“连的是 GPU 但没用 GPU”时直接点 ✕ 关掉，不要切回标准运行时（OmniVoice 必须跑 GPU）。
- 参考音频超过 10 秒时，首次生成 prompt 仍可用（20 秒实测通过）；日常复用直接加载 `verina_voice.pt`，不走参考音频。
- 只用已获授权的声音（维里奈为游戏角色配音，主人自用测试）；不要公开敏感音频。

## 文件清单

- `evening_batch.py`：晚间批量生成脚本
- `OmniVoice-Dub.ipynb`：Colab 笔记本（装包 + 跑脚本，一键）
- `ref/verina_selfintro.wav`：维里奈参考音频
- `ref/verina_selfintro.txt`：参考音频转写
- `verina_voice.pt`：声音 prompt（首次运行后生成，待补）
