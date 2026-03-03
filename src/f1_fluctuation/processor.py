import librosa
import numpy as np

class F1Vocoder:
    """声の周波数成分と位相に対して、1/fゆらぎの特性を直接調和させるクラスです。"""
    
    def __init__(self, sr: int = 44100) -> None:
        self.sr = sr

    def modulate(self, content_audio: np.ndarray, carrier_noise: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        """
        声のスペクトル（レントゲン写真）に対して、1/fの「低周波が強く、高周波が弱い」ゆらぎを付与します。
        alpha: ゆらぎの深さ（0.1〜0.2程度が、言葉が聞き取りやすく自然です）
        """
        # はみ出ないように長さを揃える
        length: int = min(len(content_audio), len(carrier_noise))
        content: np.ndarray = content_audio[:length]
        carrier: np.ndarray = carrier_noise[:length]

        # 1. 音と1/fノイズを「周波数（成分）」の世界に分解する
        stft_c = librosa.stft(content)
        stft_n = librosa.stft(carrier)

        # 声とノイズの「音の強さ（振幅）」と「波の形（位相）」を取り出す
        mag_c, phase_c = librosa.magphase(stft_c)
        mag_n, phase_n = librosa.magphase(stft_n)

        # 2. 1/fノイズから「揺れのパターン」だけを抽出する
        # ノイズの成分ごとの平均値で割り、0を中心に揺れる倍率（不規則さ）を作ります
        # ※1/fノイズを使っているので、この揺れ自体が「低周波が強く高周波が弱い」特性を持っています
        mag_n_mean = np.mean(mag_n, axis=1, keepdims=True) + 1e-8
        fluctuation = (mag_n / mag_n_mean) - 1.0 
        
        # 3. 声の強さに、1/fの揺れパターンを掛け合わせる（規則正しさと不規則さの調和）
        mag_fluctuated = mag_c * (1.0 + alpha * fluctuation)
        
        # 4. 声帯の「自然なかすれ・温かみ」を出すため、波の形（位相）にも微細な不規則さを混ぜる
        # 完全に規則的な機械音の位相を、ほんの少しだけ1/fの力で乱します
        phase_fluctuated = phase_c * np.exp(1j * (alpha * 0.1) * np.angle(phase_n))

        # 5. 揺らめかせた成分と波の形を合成し、元の時間の世界に戻す
        stft_final = mag_fluctuated * phase_fluctuated
        final_audio = librosa.istft(stft_final)

        return final_audio