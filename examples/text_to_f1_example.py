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
