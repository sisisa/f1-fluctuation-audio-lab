import numpy as np

class PinkNoiseGenerator:
    def __init__(self, sr=44100):
        self.sr = sr

    def generate_fft_method(self, duration_sec):
        """FFTベースで正確な1/fスペクトルを持つノイズを生成"""
        n_samples = int(self.sr * duration_sec)
        white_noise = np.random.randn(n_samples)
        
        # 周波数領域でのスケーリング (1/f特性)
        fft_data = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1]  # 0除算回避
        
        # 振幅を1/sqrt(f)で減衰させる
        pink_fft = fft_data / np.sqrt(freqs)
        return np.fft.irfft(pink_fft, n=n_samples)