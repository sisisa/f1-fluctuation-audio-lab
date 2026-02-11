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
