import librosa
import numpy as np
import numpy.typing as npt

class F1Vocoder:
    """音声信号に対してボコーダー処理（クロスシンセシス）を行うクラス"""
    
    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr

    def modulate(
        self, 
        content_audio: npt.NDArray[np.float32 | np.float64], 
        carrier_noise: npt.NDArray[np.float32 | np.float64]
    ) -> npt.NDArray[np.float32]:
        """
        音声の振幅特性（エンベロープ）をキャリアノイズ（1/fノイズ等）に転写する。
        
        Args:
            content_audio (npt.NDArray): 変調元となる音声データ（声など）
            carrier_noise (npt.NDArray): 搬送波となるノイズデータ
            
        Returns:
            npt.NDArray[np.float32]: 1/fゆらぎ特性が付与された合成音声
        """
        # 長さの不一致を防ぐためのアライメント処理
        length = min(len(content_audio), len(carrier_noise))
        content = content_audio[:length]
        carrier = carrier_noise[:length]

        # 短時間フーリエ変換 (STFT) による時間-周波数領域への変換
        stft_c = librosa.stft(content)
        stft_n = librosa.stft(carrier)

        # コンテンツの振幅(Magnitude)とキャリアの位相(Phase)を分離・抽出
        mag_c, _ = librosa.magphase(stft_c)
        _, phase_n = librosa.magphase(stft_n)
        
        # 振幅と位相を合成し、逆短時間フーリエ変換 (ISTFT) で時間領域へ復元
        return librosa.istft(mag_c * phase_n)

# class F1Vocoder:
#     def __init__(self, sr=44100):
#         self.sr = sr

#     def modulate(self, content_audio, carrier_noise):
#         """音声の振幅特性をノイズに転写する"""
#         # 長さ調整
#         length = min(len(content_audio), len(carrier_noise))
#         content = content_audio[:length]
#         carrier = carrier_noise[:length]

#         # STFT
#         stft_c = librosa.stft(content)
#         stft_n = librosa.stft(carrier)

#         # コンテンツの振幅(Magnitude)とキャリアの位相(Phase)を合成
#         mag_c, _ = librosa.magphase(stft_c)
#         _, phase_n = librosa.magphase(stft_n)
        
#         return librosa.istft(mag_c * phase_n)
