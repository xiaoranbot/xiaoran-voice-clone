# 在已建好的 xiaoran-trial-01 会话里跑 synthesize.py（colab exec -f 不传参数，
# 所以先把参数写进 sys.argv，再从仓库拉脚本执行）。
import sys
import urllib.request

sys.argv = [
    "synthesize.py",
    "--ref-audio-url",
    "https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/master/ref/ref_long.wav",
    "--ref-text-url",
    "https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/master/ref/ref_long.txt",
    "--gen-text",
    "官方说鱼可以留下来、只是要往后退一格，然然听着心里踏实多啦，尾巴轻轻摇着呢！",
    "--out",
    "/content/xiaoran_trial.wav",
]
code = urllib.request.urlopen(
    "https://raw.githubusercontent.com/xiaoranbot/xiaoran-voice-clone/master/synthesize.py",
    timeout=60,
).read().decode("utf-8")
exec(compile(code, "synthesize.py", "exec"))
