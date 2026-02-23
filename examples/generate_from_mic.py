import sys
import os

# プロジェクトルートをパスに追加（srcディレクトリのモジュールを読み込むため）
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from f1_fluctuation import PinkNoiseGenerator, F1Vocoder, AudioIO

def run_mic_f1_synthesis(duration=5, output_name="f1_mic_output.wav"):
    # 1. 初期化
    sr = 44100
    generator = PinkNoiseGenerator(sr=sr)
    vocoder = F1Vocoder(sr=sr)
    io = AudioIO()

    # 2. マイクから録音 
    # utils.pyのrecord_micメソッドを使用
    recorded_voice = io.record_mic(duration, sr=sr)

    # 3. 1/fノイズ（キャリア）の生成
    # 録音時間と同じ長さのピンクノイズを生成
    print(f"--- 1/fノイズ生成中 ({duration}秒) ---")
    pink_noise = generator.generate_fft_method(duration)

    # 4. ボコーダー処理（クロスシンセシス） 
    # 録音した声のエンベロープを抽出し、1/fノイズに乗せる
    print("--- 1/fゆらぎ合成処理中 ---")
    f1_voice = vocoder.modulate(recorded_voice, pink_noise)

    # 5. 結果の保存
    io.save_wav(f1_voice, output_name, sr=sr)
    print(f"完了! 自分の声が1/fゆらぎに変換されました: {output_name}")

if __name__ == "__main__":
    # 5秒間の録音と変換を実行
    run_mic_f1_synthesis(duration=5)