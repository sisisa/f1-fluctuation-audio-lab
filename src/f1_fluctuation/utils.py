import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa

class AudioIO:
    """音を録音したり、保存・読み込みをするためのお手伝いクラスです。"""
    
    @staticmethod
    def record_mic(duration: float, sr: int = 44100) -> np.ndarray:
        """指定した秒数だけ、マイクから音を録音します。"""
        print(f"--- 録音開始 ({duration}秒) ---")
        recording: np.ndarray = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()
        print("--- 録音終了 ---")
        return recording.flatten()

    @staticmethod
    def save_wav(data: np.ndarray, filename: str, sr: int = 44100) -> None:
        """出来上がった音のデータを、WAVファイルとして保存します。"""
        sf.write(filename, data, sr)
        print(f"Saved: {filename}")

    @staticmethod
    def load_wav(filename: str, sr: int = 44100) -> np.ndarray:
        """保存してあるWAVファイルを読み込みます。"""
        data, _ = librosa.load(filename, sr=sr)
        return data