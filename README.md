# f1-fluctuation-audio-lab

Python を用いて **1/f ゆらぎ（f/1 ゆらぎ）** の音声を生成し、  
- 純粋な 1/f ノイズ  
- テキストから生成した音声を 1/f ゆらぎ化した音  
- 録音した地声を 1/f ゆらぎ化した音  

を作るための実験的オーディオ DSP / AI プロジェクトです。

1/f ノイズ生成ライブラリ（例: [pyplnoise][pyplnoise]）と  
音声信号処理ツール（例: [librosa][librosa]）を組み合わせ、  
ポートフォリオとして「アルゴリズム設計〜実装〜可視化」までを一通り見せることを目的としています。

---

## Features

- ✅ Python での 1/f (pink) ノイズ生成
  - `pyplnoise.PinkNoise` を用いたパワー則ノイズ生成 [pyplnoise]
  - FFT ベースの 1/f スペクトル成形によるノイズ生成 [pink-noise-blog]

- ✅ テキスト → 1/f ゆらぎ音声
  - オフライン TTS (例: `pyttsx3`) による音声生成 [pyttsx3]
  - STFT ベースの簡易 vocoder で「声のエンベロープ × 1/f ノイズ」を合成 [librosa]

- ✅ 地声録音 → 1/f ゆらぎ音声
  - マイク録音（`sounddevice`）または既存 WAV の読み込み
  - 声のフォルマント・リズムを保ちつつ、1/f ゆらぎ質感を付与

- ✅ 分析用ノートブック
  - パワースペクトル密度 (PSD) のプロット
  - 1/f 特性の簡易検証
  - 通常音声 vs 1/f ゆらぎ音声の比較実験

---

## Project Structure

```text
f1-fluctuation-audio-lab/
├── README.md
├── requirements.txt
├── src/
│   └── f1_fluctuation/
│       ├── __init__.py
│       ├── core/
│       │   ├── noise_generators.py    # 1/f ノイズ生成（pyplnoise / FFT）
│       │   ├── vocoder.py             # STFT ベースの再合成処理
│       │   └── utils.py               # 共通ユーティリティ
│       ├── pipelines/
│       │   ├── text_to_f1.py          # テキスト → 1/f ゆらぎパイプライン
│       │   └── voice_to_f1.py         # 地声 → 1/f ゆらぎパイプライン
│       └── cli/
│           └── main.py                # CLI エントリポイント
├── examples/
│   ├── generate_pure_f1_noise.py
│   ├── text_to_f1_example.py
│   └── voice_to_f1_example.py
├── notebooks/
│   ├── 01_spectrum_analysis.ipynb
│   └── 02_vocoder_experiments.ipynb
├── data/
│   ├── input/
│   │   ├── sample_texts/
│   │   └── sample_recordings/
│   └── output/
│       ├── noise/
│       ├── text_to_f1/
│       └── voice_to_f1/
└── tests/
    ├── test_noise_generators.py
    ├── test_vocoder.py
    └── test_pipelines.py