# -*- coding: utf-8 -*-
"""
Life is what you make of it!

Written by @dinho_itt(ig_id)
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.fft import fft, ifft
from pydub import AudioSegment

# mp3 file 로딩
audio = AudioSegment.from_file(r"C:\Users\Home\Desktop\audio_data\52427187 송예화 비음 정상 발음 정상.MP3")
# mp3 파일을 모노로 변환하고 wav형식으로 변환
audio = audio.set_channels(1)
audio.export(r"C:\Users\Home\Desktop\audio_data\52427187 Phonetic normal pronunciation normal.wav", format="wav")

# wav 파일 로딩
fs, data = wavfile.read(r"C:\Users\Home\Desktop\audio_data\52427187 Phonetic normal pronunciation normal.wav")