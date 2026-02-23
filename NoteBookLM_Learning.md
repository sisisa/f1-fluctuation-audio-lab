================================================
FILE: README.md
================================================
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


================================================
FILE: f1_sample.py
================================================
import pyplnoise
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # サンプリングレート（Hz）
duration = 60  # 秒数
noisegen = pyplnoise.PinkNoise(fs, f_low=1e-3, f_high=fs/2, seed=42)  # 1/fノイズ生成器
samples = noisegen.get_series(int(fs * duration))  # 系列取得
samples = (samples * 0.5).astype(np.float32)  # 正規化（-1～1）

# 再生
sd.play(samples, fs)
sd.wait()

# WAV保存
write('1f_noise.wav', fs, (samples * 32767).astype(np.int16))


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
FILE: examples/generate_pure_f1_noise.py
================================================
"""
純粋な1/fノイズ生成例
"""

from src.f1_fluctuation.core import noise_generators, utils

if __name__ == "__main__":
    fs = 44100
    duration = 60  # 1分
    
    # pyplnoise版
    print("Generating with pyplnoise...")
    noise1 = noise_generators.generate_pink_noise(fs, duration, method="pyplnoise")
    utils.save_wav("data/output/noise/pure_f1_pyplnoise_60s.wav", noise1, fs)
    
    # FFT版
    print("Generating with FFT...")
    noise2 = noise_generators.generate_pink_noise(fs, duration, method="fft")
    utils.save_wav("data/output/noise/pure_f1_fft_60s.wav", noise2, fs)
    
    print("Done!")



================================================
FILE: examples/text_to_f1_example.py
================================================
"""
テキスト→1/fゆらぎ音声例
"""

from src.f1_fluctuation.pipelines import text_to_f1

if __name__ == "__main__":
    result = text_to_f1.text_to_f1_pipeline(
        text="ゆったりと深呼吸をしてください。リラックス...",
        output_path="data/output/text_to_f1/relax_message.wav",
        sr=22050,
        duration=15.0
    )
    print(f"Generated: {result}")



================================================
FILE: examples/voice_to_f1_example.py
================================================
"""
地声→1/fゆらぎ音声例
"""

from src.f1_fluctuation.pipelines import voice_to_f1

if __name__ == "__main__":
    # 録音モード
    print("=== Recording mode ===")
    voice_to_f1.voice_to_f1_pipeline(
        record_seconds=5.0,
        output_path="data/output/voice_to_f1/my_voice_5s.wav"
    )
    
    # ファイルモード（sample_recordingsにWAVを置く想定）
    # voice_to_f1.voice_to_f1_pipeline(
    #     input_path="data/input/sample_recordings/sample_voice.wav",
    #     output_path="data/output/voice_to_f1/sample_voice_f1.wav"
    # )



================================================
FILE: notebooks/01_spectrum_analysis.ipynb
================================================
Error processing notebook: Invalid JSON in notebook: /tmp/gitingest/7fdc78f6-1936-46b8-a673-00bfca6bd3e2/sisisa-f1-fluctuation-audio-lab/notebooks/01_spectrum_analysis.ipynb


================================================
FILE: notebooks/02_vocoder_experiments.ipynb
================================================
Error processing notebook: Invalid JSON in notebook: /tmp/gitingest/7fdc78f6-1936-46b8-a673-00bfca6bd3e2/sisisa-f1-fluctuation-audio-lab/notebooks/02_vocoder_experiments.ipynb


================================================
FILE: src/f1_fluctuation/__init__.py
================================================
"""
f1-fluctuation-audio-lab: 1/fゆらぎ音声生成ライブラリ
"""

__version__ = "0.1.0"
__author__ = "sisisa"

from .core import noise_generators, vocoder, utils
from .pipelines import text_to_f1, voice_to_f1

__all__ = [
    "noise_generators",
    "vocoder",
    "utils",
    "text_to_f1",
    "voice_to_f1"
]



================================================
FILE: src/f1_fluctuation/cli/main.py
================================================
"""
CLIエントリーポイント
"""

import argparse
import sys
from ..pipelines import text_to_f1, voice_to_f1

def main():
    parser = argparse.ArgumentParser(
        description="1/fゆらぎ音声生成ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m f1_fluctuation.cli.main text "Hello" --out result.wav
  python -m f1_fluctuation.cli.main voice --record 5 --out voice_result.wav
  python -m f1_fluctuation.cli.main voice --in input.wav --out result.wav
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # textサブコマンド
    text_parser = subparsers.add_parser("text", help="テキスト→1/fゆらぎ")
    text_parser.add_argument("text", help="変換テキスト")
    text_parser.add_argument("--out", "-o", required=True, help="出力WAV")
    text_parser.add_argument("--duration", "-d", type=float, default=10.0, help="生成時間(秒)")
    text_parser.add_argument("--noise-method", choices=["pyplnoise", "fft"], default="pyplnoise")
    
    # voiceサブコマンド
    voice_parser = subparsers.add_parser("voice", help="地声→1/fゆらぎ")
    voice_parser.add_argument("--in", "-i", help="入力WAV")
    voice_parser.add_argument("--record", type=float, help="録音秒数")
    voice_parser.add_argument("--out", "-o", required=True, help="出力WAV")
    voice_parser.add_argument("--noise-method", choices=["pyplnoise", "fft"], default="pyplnoise")
    
    args = parser.parse_args()
    
    if args.command == "text":
        result = text_to_f1.text_to_f1_pipeline(
            args.text, args.out, duration=args.duration, noise_method=args.noise_method
        )
        print(f"Generated: {result}")
    
    elif args.command == "voice":
        if not (args.input_path or args.record):
            parser.error("Specify --input or --record")

        result = voice_to_f1.voice_to_f1_pipeline(
            input_path=args.input_path,
            record_seconds=args.record,
            output_path=args.out,
            noise_method=args.noise_method
        )

        print(f"Generated: {result}")

    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()



================================================
FILE: src/f1_fluctuation/core/noise_generators.py
================================================
"""
1/fゆらぎ（ピンクノイズ）生成モジュール

- pyplnoiseを使った方法（推奨：安定・高速）
- NumPy FFTを使った方法（アルゴリズム理解用）
"""

import numpy as np
import librosa

try:
    import pyplnoise
    PYPLNOISE_AVAILABLE = True
except ImportError:
    PYPLNOISE_AVAILABLE = False
    print("Warning: pyplnoise not available. Using FFT method only.")

def generate_pink_noise_pyplnoise(fs: int, duration: float, f_low: float = 1e-3, f_high: float = None, seed: int = 42) -> np.ndarray:
    """
    pyplnoise.PinkNoiseを使った1/fノイズ生成（α=1）

    Args:
        fs: サンプリングレート (Hz)
        duration: 生成時間 (秒)
        f_low: 下限周波数
        f_high: 上限周波数 (fs/2のデフォルト)
        seed: 乱数シード

    Returns:
        正規化された1/fノイズ配列 (-1 ~ 1)
    """
    

    if f_high is None:
        f_high = fs / 2
    noisegen = pyplnoise.PinkNoise(fs, f_low=f_low, f_high=f_high, seed=seed)
    samples = noisegen.get_series(int(fs * duration))
    return librosa.util.normalize(samples.astype(np.float32))

def generate_pink_noise_fft(fs: int, duration: float, seed: int = 42) -> np.ndarray:
    """
    FFTベースの1/fノイズ生成（白色ノイズを1/fスケーリング）

    アルゴリズム:
    1. 白色ノイズを生成
    2. FFTで周波数ドメインへ
    3. |f| > 0 に対して 1/sqrt(|f|) でスケーリング
    4. 逆FFTで時間ドメインへ戻す
    """
    np.random.seed(seed)
    n_samples = int(fs * duration)
    
    # 1. 白色ノイズ生成
    white_noise = np.random.randn(n_samples)
    
    # 2. FFT (実数FFT)
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    white_fft = np.fft.rfft(white_noise)
    
    # 3. 1/fスケーリング (DC成分(f=0)は除く)
    pink_fft = white_fft.copy()
    scale = np.sqrt(1.0 / (freqs + 1e-10))  # 1/sqrt(f) で振幅スケール
    pink_fft[1:] *= scale[1:]  # DC成分はスケールしない
    
    # 4. 逆FFT
    pink_noise = np.fft.irfft(pink_fft, n=n_samples)
    
    return librosa.util.normalize(pink_noise.astype(np.float32))

def generate_pink_noise(fs: int, duration: float, method: str = "pyplnoise", **kwargs) -> np.ndarray:
    """
    1/fノイズ生成のラッパー関数
    
    Args:
        method: "pyplnoise" or "fft"
    """
    if method == "pyplnoise" and PYPLNOISE_AVAILABLE:
        return generate_pink_noise_pyplnoise(fs, duration, **kwargs)
    else:
        print("Using FFT method (pyplnoise not available)")
        return generate_pink_noise_fft(fs, duration, **kwargs)



================================================
FILE: src/f1_fluctuation/core/utils.py
================================================
"""
共通ユーティリティ
"""

import numpy as np
import librosa
from scipy.io import wavfile
import os

def ensure_dir(directory: str):
    """ディレクトリが存在しなければ作成"""
    os.makedirs(directory, exist_ok=True)

def save_wav(filename: str, audio: np.ndarray, sr: int):
    """WAV保存（int16）"""
    ensure_dir(os.path.dirname(filename))
    wavfile.write(filename, sr, (audio * 32767).astype(np.int16))
    print(f"Saved: {filename}")

def load_audio(filename: str, sr: int = None) -> tuple:
    """音声読み込み（正規化）"""
    audio, file_sr = librosa.load(filename, sr=sr)
    return librosa.util.normalize(audio), file_sr

def plot_spectrum(audio: np.ndarray, sr: int, title: str = "PSD"):
    """簡易PSDプロット（matplotlib使用）"""
    import matplotlib.pyplot as plt
    
    f, Pxx = librosa.psd(audio, sr=sr)
    plt.semilogx(f, 10 * np.log10(Pxx))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB]')
    plt.title(f'{title} (1/f slope expected)')
    plt.grid(True)
    plt.show()



================================================
FILE: src/f1_fluctuation/core/vocoder.py
================================================
"""
STFTベースの簡易Vocoder: 入力音声のエンベロープをノイズで再合成
"""

import numpy as np
import librosa

def f1_vocoder(input_audio: np.ndarray, carrier_noise: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    1/fゆらぎVocoder: 入力音声の振幅スペクトル + キャリアノイズの位相で再合成

    アルゴリズム:
    1. 両信号をSTFT
    2. |input_stft| * exp(j * angle(carrier_stft))
    3. iSTFTで時間波形へ戻す

    Args:
        input_audio: 元音声（TTS or 録音）
        carrier_noise: 1/fノイズ（同長）
        sr: サンプリングレート
        hop_length: STFTのhop length

    Returns:
        1/fゆらぎ化した音声
    """
    # STFT (入力音声とキャリア)
    stft_input = librosa.stft(input_audio, hop_length=hop_length)
    stft_carrier = librosa.stft(carrier_noise, hop_length=hop_length)
    
    # Vocoder合成: 入力の振幅 + キャリアの位相
    modulated_stft = np.abs(stft_input) * np.exp(1j * np.angle(stft_carrier))
    
    # iSTFT
    modulated_audio = librosa.istft(modulated_stft, hop_length=hop_length)
    
    return librosa.util.normalize(modulated_audio)

def simple_phase_vocoder_swap(input_audio: np.ndarray, carrier_noise: np.ndarray, sr: int) -> np.ndarray:
    """
    librosa.phase_vocoderを使った簡易版（学習用）
    """
    return librosa.phase_vocoder(carrier_noise, sr=sr, rate=1.0)  # 簡易実装



================================================
FILE: src/f1_fluctuation/pipelines/text_to_f1.py
================================================
[Binary file]


================================================
FILE: src/f1_fluctuation/pipelines/voice_to_f1.py
================================================
[Binary file]


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

