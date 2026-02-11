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
