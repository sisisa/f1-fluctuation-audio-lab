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
