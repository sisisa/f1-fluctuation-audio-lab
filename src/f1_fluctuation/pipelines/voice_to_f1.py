"""
地声録音 → 1/fゆらぎ音声パイプライン
"""

from ..core import noise_generators, vocoder, utils
import librosa

def voice_to_f1_pipeline(
    output_path: str,
    input_path: str | None = None,
    record_seconds: float | None = None,
    sr: int = 22050,
    noise_method: str = "pyplnoise"
) -> str:
    """
    地声 → 1/fゆらぎ音声
    
    Args:
        input_path: 入力WAVパス（Noneなら録音）
        record_seconds: 録音秒数（input_path=None時）
        output_path: 出力WAVパス
        sr, noise_method: 同上
    """
    if input_path:
        print(f"Loading voice from {input_path}")
        voice_audio, _ = utils.load_audio(input_path, sr=sr)
    else:
        print(f"Recording {record_seconds}s voice...")
        import sounddevice as sd
        voice_audio = sd.rec(
            int(record_seconds * sr), samplerate=sr, channels=1
        ).flatten()
        sd.wait()
        voice_audio = librosa.util.normalize(voice_audio)
    
    # 1/fノイズ生成（同長）
    noise_duration = len(voice_audio) / sr
    f1_noise = noise_generators.generate_pink_noise(
        sr, noise_duration, method=noise_method
    )
    
    # Vocoder合成
    print("Applying 1/f vocoder to voice...")
    f1_voice = vocoder.f1_vocoder(voice_audio, f1_noise, sr)
    
    # 保存
    utils.save_wav(output_path, f1_voice, sr)
    return output_path
