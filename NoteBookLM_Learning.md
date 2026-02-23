================================================
FILE: README.md
================================================
[Binary file]


================================================
FILE: f1_sample.py
================================================
import numpy as np
import librosa
import soundfile as sf
import pyttsx3

class F1AudioLab:
    def __init__(self, sr=44100):
        self.sr = sr

    """FFTベースで1/fスペクトルを持つピンクノイズを生成"""
    def generate_pink_noise(self, duration):
        n_samples = int(self.sr * duration)
        # 白色ノイズを生成
        white_noise = np.random.randn(n_samples)
        
        # 周波数領域に変換
        f = np.fft.rfftfreq(n_samples)
        f[0] = 1  # 0除算回避
        
        # 1/f 特性（振幅スペクトルでは 1/sqrt(f)）を適用
        scaler = 1 / np.sqrt(f)
        fft_white = np.fft.rfft(white_noise)
        fft_pink = fft_white * scaler
        
        # 時間領域に戻す
        pink_noise = np.fft.irfft(fft_pink, n=n_samples)
        return pink_noise / np.max(np.abs(pink_noise))

    """音声のエンベロープを抽出し、1/fノイズに適用する(簡易Vocoder)"""
    def apply_f1_to_voice(self, voice_data, pink_noise):        
        # 長さを合わせる
        min_len = min(len(voice_data), len(pink_noise))
        voice = voice_data[:min_len]
        noise = pink_noise[:min_len]

        # 短時間フーリエ変換 (STFT)
        stft_voice = librosa.stft(voice)
        stft_noise = librosa.stft(noise)

        # 声のスペクトル包絡（振幅）のみを抽出
        magnitude_voice, _ = librosa.magphase(stft_voice)
        # ノイズの位相を取得
        _, phase_noise = librosa.magphase(stft_noise)

        # 声の大きさをノイズに乗せる
        combined_stft = magnitude_voice * phase_noise
        
        # 時間領域に復元
        return librosa.istft(combined_stft)

    """テキストからTTS音声を生成し、1/f加工を施す"""
    def text_to_f1_speech(self, text, output_file="f1_speech.wav"):
        engine = pyttsx3.init()
        temp_file = "temp_tts.wav"
        engine.save_to_file(text, temp_file)
        engine.runAndWait()

        # 生成した音声を読み込み
        voice, _ = librosa.load(temp_file, sr=self.sr)
        pink_noise = self.generate_pink_noise(len(voice) / self.sr)
        
        f1_voice = self.apply_f1_to_voice(voice, pink_noise)
        sf.write(output_file, f1_voice, self.sr)
        print(f"Saved: {output_file}")

# 実行例
if __name__ == "__main__":
    lab = F1AudioLab()
    
    # 1. 純粋なピンクノイズの生成
    pink = lab.generate_pink_noise(3.0)
    sf.write("pure_pink_noise.wav", pink, 44100)
    
    # 2. テキストからの1/fゆらぎ音声生成
    lab.text_to_f1_speech("1/fゆらぎの世界へようこそ。")



================================================
FILE: NoteBookLM_Learning.md
================================================
[Binary file]


================================================
FILE: requirements.txt
================================================
numpy
scipy
librosa
pyplnoise
pyttsx3
sounddevice



================================================
FILE: examples/generate_from_mic.py
================================================
[Empty file]


================================================
FILE: examples/generate_from_text.py
================================================
[Binary file]


================================================
FILE: notebooks/01_spectrum_analysis.ipynb
================================================
Error processing notebook: Invalid JSON in notebook: /tmp/gitingest/2f1ad97a-903b-4b2d-a8ef-63a652466d05/sisisa-f1-fluctuation-audio-lab/notebooks/01_spectrum_analysis.ipynb


================================================
FILE: notebooks/02_vocoder_experiments.ipynb
================================================
Error processing notebook: Invalid JSON in notebook: /tmp/gitingest/2f1ad97a-903b-4b2d-a8ef-63a652466d05/sisisa-f1-fluctuation-audio-lab/notebooks/02_vocoder_experiments.ipynb


================================================
FILE: src/f1_fluctuation/__init__.py
================================================
"""
f1-fluctuation-audio-lab: 1/fゆらぎ音声生成ライブラリ
"""

from .generator import PinkNoiseGenerator
from .processor import F1Vocoder
from .utils import AudioIO

__all__ = ["PinkNoiseGenerator", "F1Vocoder", "AudioIO"]


================================================
FILE: src/f1_fluctuation/generator.py
================================================
import numpy as np

class PinkNoiseGenerator:
    def __init__(self, sr=44100):
        self.sr = sr

    def generate_fft_method(self, duration_sec):
        """FFTベースで正確な1/fスペクトルを持つノイズを生成"""
        n_samples = int(self.sr * duration_sec)
        white_noise = np.random.randn(n_samples)
        
        # 周波数領域でのスケーリング (1/f特性)
        fft_data = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(n_samples)
        freqs[0] = freqs[1]  # 0除算回避
        
        # 振幅を1/sqrt(f)で減衰させる
        pink_fft = fft_data / np.sqrt(freqs)
        return np.fft.irfft(pink_fft, n=n_samples)


================================================
FILE: src/f1_fluctuation/processor.py
================================================
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


================================================
FILE: src/f1_fluctuation/utils.py
================================================
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


================================================
FILE: tests/test_noise_generators.py
================================================
[Empty file]


================================================
FILE: tests/test_pipelines.py
================================================
[Empty file]


================================================
FILE: tests/test_vocoder.py
================================================
[Empty file]

