import sys
import os
import json
import glob

# 自分の作った「ゆらぎの道具箱」を使えるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# 先ほど作った generate_from_text.py の中から、音声合成の機能だけを呼び出す
from generate_from_text import run_podcast_synthesis

def process_all_contents() -> None:
    # 注文ポスト（contentsフォルダ）の場所を指定
    contents_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'contents')
    
    # フォルダの中にある「.json」で終わる注文書をすべて探す
    json_files = glob.glob(os.path.join(contents_dir, "*.json"))

    if not json_files:
        print("注文書（JSONファイル）が見つかりません。src/contents/ フォルダを確認してください。")
        return

    print(f"--- 合計 {len(json_files)} 件の注文書が見つかりました ---")

    # 見つけた注文書を1つずつ開いて、順番に音声を作っていく
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # 注文書を読み込む
        
        # 注文書に書かれているデータを取り出す
        text = data.get("text", "")
        alpha = data.get("alpha", 0.3)
        output_name = data.get("output_name", "output.wav")

        print(f"\n▶ 処理開始: {os.path.basename(file_path)}")
        
        # もしテキストが空っぽなら飛ばす
        if not text:
            print("エラー: 台本（text）が空のためスキップします。")
            continue

        # 用意してある「ポッドキャスト作成機能」に注文データを渡して実行！
        run_podcast_synthesis(text, output_name=output_name, alpha=alpha)

    print("\nすべての注文の処理が完了しました！")

if __name__ == "__main__":
    process_all_contents()