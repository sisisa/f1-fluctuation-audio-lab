"""
1/fゆらぎ（ピンクノイズ）生成モジュール

- pyplnoiseを使った方法（推奨：安定・高速）
- NumPy FFTを使った方法（アルゴリズム理解用）
"""

import numpy as np
import librosa

try:
    import pyplnoise
    PYPLNOISE_AVAILABLE = True
except ImportError:
    PYPLNOISE_AVAILABLE = False
    print("Warning: pyplnoise not available. Using FFT method only.")

def generate_pink_noise_pyplnoise(fs: int, duration: float, f_low: float = 1e-3, f_high: float = None, seed: int = 42) -> np.ndarray:
    """
    pyplnoise.PinkNoiseを使った1/fノイズ生成（α=1）

    Args:
        fs: サンプリングレート (Hz)
        duration: 生成時間 (秒)
        f_low: 下限周波数
        f_high: 上限周波数 (fs/2のデフォルト)
        seed: 乱数シード

    Returns:
        正規化された1/fノイズ配列 (-1 ~ 1)
    """
    

    if f_high is None:
        f_high = fs / 2
    noisegen = pyplnoise.PinkNoise(fs, f_low=f_low, f_high=f_high, seed=seed)
    samples = noisegen.get_series(int(fs * duration))
    return librosa.util.normalize(samples.astype(np.float32))

def generate_pink_noise_fft(fs: int, duration: float, seed: int = 42) -> np.ndarray:
    """
    FFTベースの1/fノイズ生成（白色ノイズを1/fスケーリング）

    アルゴリズム:
    1. 白色ノイズを生成
    2. FFTで周波数ドメインへ
    3. |f| > 0 に対して 1/sqrt(|f|) でスケーリング
    4. 逆FFTで時間ドメインへ戻す
    """
    np.random.seed(seed)
    n_samples = int(fs * duration)
    
    # 1. 白色ノイズ生成
    white_noise = np.random.randn(n_samples)
    
    # 2. FFT (実数FFT)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    white_fft = np.fft.rfft(white_noise)
    
    # 3. 1/fスケーリング (DC成分(f=0)は除く)
    pink_fft = white_fft.copy()
    scale = np.sqrt(1.0 / (freqs + 1e-10))  # 1/sqrt(f) で振幅スケール
    pink_fft[1:] *= scale[1:]  # DC成分はスケールしない
    
    # 4. 逆FFT
    pink_noise = np.fft.irfft(pink_fft, n=n_samples)
    
    return librosa.util.normalize(pink_noise.astype(np.float32))

def generate_pink_noise(fs: int, duration: float, method: str = "pyplnoise", **kwargs) -> np.ndarray:
    """
    1/fノイズ生成のラッパー関数
    
    Args:
        method: "pyplnoise" or "fft"
    """
    if method == "pyplnoise" and PYPLNOISE_AVAILABLE:
        return generate_pink_noise_pyplnoise(fs, duration, **kwargs)
    else:
        print("Using FFT method (pyplnoise not available)")
        return generate_pink_noise_fft(fs, duration, **kwargs)
