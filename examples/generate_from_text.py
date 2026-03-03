import sys
import os
import numpy as np
import argparse
from gtts import gTTS

# 自分の作った「ゆらぎ生成の設定」を使えるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from f1_fluctuation.generator import PinkNoiseGenerator
from f1_fluctuation.processor import F1Vocoder
from f1_fluctuation.utils import AudioIO

def run_podcast_synthesis(text: str, output_name: str = "f1_podcast.wav", alpha: float = 0.3) -> None:
    # 1. 道具の準備
    sr: int = 44100
    generator = PinkNoiseGenerator(sr=sr)
    vocoder = F1Vocoder(sr=sr)
    io = AudioIO()

    # 2. 大きなお肉（長文）を、一口サイズ（文ごと）に切り分ける
    sentences = text.split("。")
    final_audio_pieces = [] # 出来上がった音を貯めておくお皿

    print(f"--- ポッドキャスト生成開始（ゆらぎの強さ: {alpha}） ---")

    for index, sentence in enumerate(sentences):
        # 空っぽの文は飛ばす
        if not sentence.strip():
            continue
            
        sentence_text = sentence + "。"
        print(f"処理中 ({index+1}): {sentence_text}")

        # 一口サイズごとに音声を生成・読み込み
        temp_audio = f"temp_tts_{index}.mp3"
        tts = gTTS(text=sentence_text, lang='ja')
        tts.save(temp_audio)
        content_audio = io.load_wav(temp_audio, sr=sr)
        duration: float = len(content_audio) / sr

        # 一口サイズごとにゆらぎを作って混ぜる（alphaで味付け調整）
        pink_noise = generator.generate_fft_method(duration)
        f1_voice = vocoder.modulate(content_audio, pink_noise, alpha=alpha)

        # 出来上がった音をお皿に乗せ、ゴミ（一時ファイル）を捨てる
        final_audio_pieces.append(f1_voice)
        os.remove(temp_audio)

    # 3. お皿に乗った一口サイズの音を、全てくっつける
    print("--- 音声をつなぎ合わせています ---")
    combined_audio = np.concatenate(final_audio_pieces)

    # 4. 完成品の保存
    io.save_wav(combined_audio, output_name, sr=sr)
    print(f"完了! 完成したファイル: {output_name}")

if __name__ == "__main__":
    # ターミナルから命令を受け取るための「受付窓口」を作ります
    parser = argparse.ArgumentParser(description="テキストから1/fゆらぎポッドキャスト音声を生成します。")
    
    # 受け付けるボタン（引数）の種類を決めます
    parser.add_argument("--text", type=str, help="読み上げたい文章を直接入力します。")
    parser.add_argument("--file", type=str, help="読み上げたい文章が書かれたテキストファイルを指定します。")
    parser.add_argument("--alpha", type=float, default=0.3, help="ゆらぎの強さを調整します（0.0〜1.0）。デフォルトは0.3です。")
    parser.add_argument("--out", type=str, default="f1_podcast.wav", help="保存するファイル名を指定します。")

    args = parser.parse_args()

    # 読み上げる文章を準備する
    content_text = ""
    
    # ファイルが指定されたら、その中身を読む
    if args.file and os.path.exists(args.file):
        with open(args.file, 'r', encoding='utf-8') as f:
            content_text = f.read()
    # テキストが直接入力されたら、それを使う
    elif args.text:
        content_text = args.text
    # どちらもなければ、エラーを出して終わる
    else:
        print("エラー: --text で文章を入力するか、 --file でテキストファイルを指定してください。")
        sys.exit(1)

    # 用意した文章と設定で、音声を生成する
    run_podcast_synthesis(content_text, output_name=args.out, alpha=args.alpha)
if __name__ == "__main__":
    # ポッドキャスト用の長い文章
    sample_text = (
        "皆さん、こんばんは。今日の出来事はいかがでしたか。"
        "少し疲れたという方も、とても楽しかったという方もいるでしょう。"
        "この時間は、自然のゆらぎを感じながら、ゆっくりと心を休めてください。"
        "明日があなたにとって、素晴らしい一日になりますように。"
    )
    
    # ゆらぎの強さ（alpha）を0.3にして実行する
    run_podcast_synthesis(sample_text, alpha=0.3)