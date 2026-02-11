"""
STFTベースの簡易Vocoder: 入力音声のエンベロープをノイズで再合成
"""

import numpy as np
import librosa

def f1_vocoder(input_audio: np.ndarray, carrier_noise: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    1/fゆらぎVocoder: 入力音声の振幅スペクトル + キャリアノイズの位相で再合成

    アルゴリズム:
    1. 両信号をSTFT
    2. |input_stft| * exp(j * angle(carrier_stft))
    3. iSTFTで時間波形へ戻す

    Args:
        input_audio: 元音声（TTS or 録音）
        carrier_noise: 1/fノイズ（同長）
        sr: サンプリングレート
        hop_length: STFTのhop length

    Returns:
        1/fゆらぎ化した音声
    """
    # STFT (入力音声とキャリア)
    stft_input = librosa.stft(input_audio, hop_length=hop_length)
    stft_carrier = librosa.stft(carrier_noise, hop_length=hop_length)
    
    # Vocoder合成: 入力の振幅 + キャリアの位相
    modulated_stft = np.abs(stft_input) * np.exp(1j * np.angle(stft_carrier))
    
    # iSTFT
    modulated_audio = librosa.istft(modulated_stft, hop_length=hop_length)
    
    return librosa.util.normalize(modulated_audio)

def simple_phase_vocoder_swap(input_audio: np.ndarray, carrier_noise: np.ndarray, sr: int) -> np.ndarray:
    """
    librosa.phase_vocoderを使った簡易版（学習用）
    """
    return librosa.phase_vocoder(carrier_noise, sr=sr, rate=1.0)  # 簡易実装
