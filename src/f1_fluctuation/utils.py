import sounddevice as sd
import soundfile as sf
import numpy as np
import numpy.typing as npt
import librosa
from pathlib import Path
from typing import Union

class AudioIO:
    """音声の入出力および録音を管理するユーティリティクラス"""
    
    @staticmethod
    def record_mic(duration: float, sr: int = 44100) -> npt.NDArray[np.float32]:
        """
        指定された秒数、マイクから音声を録音する。
        
        Raises:
            sd.SoundDeviceError: オーディオデバイスへのアクセスに失敗した場合
        """
        if duration <= 0:
            raise ValueError("録音時間は0より大きい必要があります。")
            
        print(f"--- 録音開始 ({duration}秒) ---")
        try:
            # 1チャンネル(モノラル)で録音
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()  # 録音終了までブロック
            print("--- 録音終了 ---")
            return recording.flatten()
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def save_wav(data: npt.NDArray[np.float32 | np.float64], filename: Union[str, Path], sr: int = 44100) -> None:
        """
        音声データをWAVファイルとして安全に保存する。
        """
        file_path = Path(filename)
        # 保存先ディレクトリが存在しない場合は作成を試みるなどの拡張も可能
        try:
            sf.write(str(file_path), data, sr)
            print(f"Saved: {file_path.absolute()}")
        except Exception as e:
            print(f"ファイルの保存に失敗しました: {e}")
            raise

    @staticmethod
    def load_wav(filename: Union[str, Path], sr: int = 44100) -> npt.NDArray[np.float32]:
        """
        既存のWAVファイルを指定したサンプリングレートで読み込む。
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
        try:
            data, _ = librosa.load(str(file_path), sr=sr)
            return data
        except Exception as e:
            print(f"ファイルの読み込みに失敗しました: {e}")
            raise
            
# """音声の入出力および録音を管理するユーティリティクラス"""
# class AudioIO:
#     @staticmethod
#     def record_mic(duration, sr=44100):
#         """
#         指定された秒数、マイクから音声を録音する
#         """
#         print(f"--- 録音開始 ({duration}秒) ---")
#         recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
#         sd.wait()  # 録音終了まで待機
#         print("--- 録音終了 ---")
#         return recording.flatten()

#     @staticmethod
#     def save_wav(data, filename, sr=44100):
#         """
#         音声データをWAVファイルとして保存する
#         """
#         sf.write(filename, data, sr)
#         print(f"Saved: {filename}")

#     @staticmethod
#     def load_wav(filename, sr=44100):
#         """
#         既存のWAVファイルを読み込む 
#         """
#         data, _ = librosa.load(filename, sr=sr)
#         return data
