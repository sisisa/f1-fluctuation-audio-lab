import librosa
import numpy as np

class F1Vocoder:
    def __init__(self, sr=44100):
        self.sr = sr

    def modulate(self, content_audio, carrier_noise):
        """音声の振幅特性をノイズに転写する"""
        # 長さ調整
        length = min(len(content_audio), len(carrier_noise))
        content = content_audio[:length]
        carrier = carrier_noise[:length]

        # STFT
        stft_c = librosa.stft(content)
        stft_n = librosa.stft(carrier)

        # コンテンツの振幅(Magnitude)とキャリアの位相(Phase)を合成
        mag_c, _ = librosa.magphase(stft_c)
        _, phase_n = librosa.magphase(stft_n)
        
        return librosa.istft(mag_c * phase_n)