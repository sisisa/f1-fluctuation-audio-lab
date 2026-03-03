import numpy as np

class PinkNoiseGenerator:
    """1/fゆらぎ（ピンクノイズ）を作るための専用のクラスです。"""
    
    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr

    def generate_fft_method(self, duration_sec: float) -> np.ndarray:
        """
        指定された秒数のピンクノイズを作ります。
        
        工夫した点:
        感覚で作るのではなく、周波数の計算を用いて
        数学的に正確な「1/f」のゆらぎを作り出しています。
        """
        n_samples: int = int(self.sr * duration_sec)
        white_noise: np.ndarray = np.random.randn(n_samples)
        
        # 周波数の世界に変換して、1/fのルールを当てはめる
        fft_data: np.ndarray = np.fft.rfft(white_noise)
        freqs: np.ndarray = np.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1]  # 0で割るエラーを防ぐための工夫
        
        # 波の大きさを調整して、元の時間の世界に戻す
        pink_fft: np.ndarray = fft_data / np.sqrt(freqs)
        return np.fft.irfft(pink_fft, n=n_samples)