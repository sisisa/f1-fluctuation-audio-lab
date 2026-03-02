import numpy as np
import numpy.typing as npt

class PinkNoiseGenerator:
    """1/fゆらぎ（ピンクノイズ）を生成するクラス"""
    def __init__(self, sr: int = 44100) -> None:
        if sr <= 0:
            raise ValueError("サンプリングレート(sr)は正の整数である必要があります。")
        self.sr = sr

    def generate_fft_method(self, duration_sec: float) -> npt.NDArray[np.float64]:
        """
        FFTベースで正確な1/fスペクトルを持つノイズを生成する。
        
        Args:
            duration_sec (float): 生成するノイズの長さ（秒）
            
        Returns:
            npt.NDArray[np.float64]: 生成されたピンクノイズの1次元配列
            
        Raises:
            ValueError: duration_secが0以下の場合
        """
        if duration_sec <= 0:
            raise ValueError("生成時間(duration_sec)は0より大きい必要があります。")

        n_samples = int(self.sr * duration_sec)
        white_noise = np.random.randn(n_samples)
        
        # 周波数領域でのスケーリング (1/f特性)
        fft_data = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples)
        
        # 0Hz成分（直流成分）の0除算を回避し、低周波の無限大発散を防ぐ
        freqs[0] = freqs[1]
        
        # 振幅を1/sqrt(f)で減衰させることで、パワースペクトル密度(PSD)を1/fにする
        pink_fft = fft_data / np.sqrt(freqs)
        pink_noise = np.fft.irfft(pink_fft, n=n_samples)
        
        # クリッピング防止のための正規化
        return pink_noise / np.max(np.abs(pink_noise))


# class PinkNoiseGenerator:
#     def __init__(self, sr=44100):
#         self.sr = sr

#     def generate_fft_method(self, duration_sec):
#         """FFTベースで正確な1/fスペクトルを持つノイズを生成"""
#         n_samples = int(self.sr * duration_sec)
#         white_noise = np.random.randn(n_samples)
        
#         # 周波数領域でのスケーリング (1/f特性)
#         fft_data = np.fft.rfft(white_noise)
#         freqs = np.fft.rfftfreq(n_samples)
#         freqs[0] = freqs[1]  # 0除算回避
        
#         # 振幅を1/sqrt(f)で減衰させる
#         pink_fft = fft_data / np.sqrt(freqs)
#         return np.fft.irfft(pink_fft, n=n_samples)
