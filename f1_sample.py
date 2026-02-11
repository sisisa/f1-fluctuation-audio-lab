import pyplnoise
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # サンプリングレート（Hz）
duration = 60  # 秒数
noisegen = pyplnoise.PinkNoise(fs, f_low=1e-3, f_high=fs/2, seed=42)  # 1/fノイズ生成器
samples = noisegen.get_series(int(fs * duration))  # 系列取得
samples = (samples * 0.5).astype(np.float32)  # 正規化（-1～1）

# 再生
sd.play(samples, fs)
sd.wait()

# WAV保存
write('1f_noise.wav', fs, (samples * 32767).astype(np.int16))