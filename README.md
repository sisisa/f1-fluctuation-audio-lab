# f1-fluctuation-audio-lab

1/fゆらぎ（ピンクノイズ）を音声信号処理に応用し、癒やしの質感を付与する実験的DSPプロジェクト。

## 設計思想
本プロジェクトは、自然界に存在する「1/fゆらぎ」をデジタル音声に統合することを目的にしています。
1. **数学的生成**: パワー則 $S(f) \propto 1/f$ に基づく厳密なノイズ生成。
2. **クロスシンセシス**: 音声のフォルマントと1/fノイズをSTFT領域で結合。
3. **客観的検証**: PSD（パワースペクトル密度）による特性の可視化。

## 使い方
### 1. 環境構築
```bash
pip install -r requirements.txt
```

## Project Structure

```text

f1-fluctuation-audio-lab/
├── README.md              # プロジェクト概要と使用方法（改良版）
├── requirements.txt       # 依存ライブラリ
├── src/
│   └── f1_fluctuation/
│       ├── __init__.py
│       ├── generator.py   # 1/fノイズ生成アルゴリズム
│       ├── processor.py   # 音声合成・ボコーダー処理
│       └── utils.py       # ファイル入出力・録音ユーティリティ
├── examples/
│   ├── generate_from_text.py  # TTS連携サンプル
│   └── generate_from_mic.py   # 録音連携サンプル
├── notebooks/
│   └── analysis_psd.ipynb     # パワースペクトル密度（PSD）分析
└── tests/
    └── test_generator.py      # ユニットテスト

```