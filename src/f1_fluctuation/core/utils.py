"""
共通ユーティリティ
"""

import numpy as np
import librosa
from scipy.io import wavfile
import os

def ensure_dir(directory: str):
    """ディレクトリが存在しなければ作成"""
    os.makedirs(directory, exist_ok=True)

def save_wav(filename: str, audio: np.ndarray, sr: int):
    """WAV保存（int16）"""
    ensure_dir(os.path.dirname(filename))
    wavfile.write(filename, sr, (audio * 32767).astype(np.int16))
    print(f"Saved: {filename}")

def load_audio(filename: str, sr: int = None) -> tuple:
    """音声読み込み（正規化）"""
    audio, file_sr = librosa.load(filename, sr=sr)
    return librosa.util.normalize(audio), file_sr

def plot_spectrum(audio: np.ndarray, sr: int, title: str = "PSD"):
    """簡易PSDプロット（matplotlib使用）"""
    import matplotlib.pyplot as plt
    
    f, Pxx = librosa.psd(audio, sr=sr)
    plt.semilogx(f, 10 * np.log10(Pxx))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB]')
    plt.title(f'{title} (1/f slope expected)')
    plt.grid(True)
    plt.show()
