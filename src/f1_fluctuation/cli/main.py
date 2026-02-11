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
