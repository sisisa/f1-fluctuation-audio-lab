import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa

"""音声の入出力および録音を管理するユーティリティクラス"""
class AudioIO:
    @staticmethod
    def record_mic(duration, sr=44100):
        """
        指定された秒数、マイクから音声を録音する
        """
        print(f"--- 録音開始 ({duration}秒) ---")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # 録音終了まで待機
        print("--- 録音終了 ---")
        return recording.flatten()

    @staticmethod
    def save_wav(data, filename, sr=44100):
        """
        音声データをWAVファイルとして保存する
        """
        sf.write(filename, data, sr)
        print(f"Saved: {filename}")

    @staticmethod
    def load_wav(filename, sr=44100):
        """
        既存のWAVファイルを読み込む 
        """
        data, _ = librosa.load(filename, sr=sr)
        return data