import sys
import os
import pyttsx3

# プロジェクトルートをパスに追加（srcディレクトリのモジュールを読み込むため）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from f1_fluctuation import PinkNoiseGenerator, F1Vocoder, AudioIO

def run_tts_f1_synthesis(text, output_name="f1_output.wav"):
    # 1. 初期化
    sr = 44100
    generator = PinkNoiseGenerator(sr=sr)
    vocoder = F1Vocoder(sr=sr)
    io = AudioIO()

    print(f"--- TTS生成開始: '{text}' ---")
    
    # 2. TTS (Text-to-Speech) で一時的な音声を生成 
    engine = pyttsx3.init()
    temp_wav = "temp_tts.wav"
    engine.save_to_file(text, temp_wav)
    engine.runAndWait()

    # 3. 音声の読み込み
    content_audio = io.load_wav(temp_wav, sr=sr)
    duration = len(content_audio) / sr

    # 4. 1/fノイズ（キャリア）の生成 
    print(f"--- 1/fノイズ生成中 ({duration:.2f}秒) ---")
    pink_noise = generator.generate_fft_method(duration)

    # 5. ボコーダー処理（声のエンベロープ × 1/fノイズ） 
    print("--- 1/fゆらぎ合成処理中 ---")
    f1_voice = vocoder.modulate(content_audio, pink_noise)

    # 6. 保存と後処理
    io.save_wav(f1_voice, output_name, sr=sr)
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    print(f"完了! 生成ファイル: {output_name}")

if __name__ == "__main__":
    sample_text = "こんにちは。これは、1/fゆらぎを付与した合成音声のテストです。リラックス効果を期待しています。"
    run_tts_f1_synthesis(sample_text)