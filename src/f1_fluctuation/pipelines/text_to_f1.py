"""
テキスト → 1/fゆらぎ音声パイプライン
"""

import pyttsx3
from ..core import noise_generators, vocoder, utils

def text_to_f1_pipeline(
    text: str,
    output_path: str,
    sr: int = 22050,
    duration: float = 10.0,
    noise_method: str = "pyplnoise"
) -> str:
    """
    テキストから1/fゆらぎ音声までを一気通貫で実行
    
    Args:
        text: 変換対象テキスト
        output_path: 出力WAVパス
        sr: サンプリングレート
        duration: ノイズ長（秒）
        noise_method: "pyplnoise" or "fft"
    
    Returns:
        出力ファイルパス
    """
    print(f"Step 1: TTS '{text}'...")
    
    # 1. TTS生成（一時ファイル）
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    tts_temp = "temp_tts.wav"
    engine.save_to_file(text, tts_temp)
    engine.runAndWait()
    
    # 2. TTS読み込み
    tts_audio, _ = utils.load_audio(tts_temp, sr=sr)
    
    # 3. 1/fノイズ生成（TTSと同長）
    noise_duration = len(tts_audio) / sr
    f1_noise = noise_generators.generate_pink_noise(
        sr, noise_duration, method=noise_method
    )
    
    # 4. Vocoder合成
    print("Step 2: Applying 1/f vocoder...")
    f1_voice = vocoder.f1_vocoder(tts_audio, f1_noise, sr)
    
    # 5. 保存
    utils.save_wav(output_path, f1_voice, sr)
    
    # 一時ファイル削除
    import os
    os.remove(tts_temp)
    
    return output_path
