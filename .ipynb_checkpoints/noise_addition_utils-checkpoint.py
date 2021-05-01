import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors, pyplot as plt
from numpy.fft import rfft, irfft

from scipy.io.wavfile import read

import torchaudio


# Set Audio backend as Sounfile for windows and Sox for Linux
torchaudio.set_audio_backend("soundfile")


def ms(x):
    return (np.abs(x)**2).mean()


def rms(x):
    return np.sqrt(ms(x))


def normalise(y, power):
    """
    Normalise power in y to power specified.
    Standard signal if power=1
    The mean power of a Gaussian with `mu=0` and `sigma=x` is x^2.
    """
    return y * np.sqrt(power / ms(y))


def noise(N, color, power):
    """
    Noise generator.
    N: Amount of samples.
    color: Color of noise.
    power: power = std_dev^2
    https://en.wikipedia.org/wiki/Colors_of_noise
    """
    noise_generators = {
        'white': white,
        'pink': pink,
        'blue': blue,
        'brown': brown,
        'violet': violet
    }
    return noise_generators[color](N, power)


def white(N, power):
    y = np.random.randn(N).astype(np.float32)
    return normalise(y, power)


def pink(N, power):
    orig_N = N
    # Because doing rfft->ifft produces different length outputs depending if its odd or even length inputs
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size)+1.)  # +1 to avoid divide by zero
    y = irfft(X/S).real[:orig_N]
    return normalise(y, power)


def blue(N, power):
    orig_N = N
    # Because doing rfft->ifft produces different length outputs depending if its odd or even length inputs
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.sqrt(np.arange(X.size))  # Filter
    y = irfft(X*S).real[:orig_N]
    return normalise(y, power)


def brown(N, power):
    orig_N = N
    # Because doing rfft->ifft produces different length outputs depending if its odd or even length inputs
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size)+1  # Filter
    y = irfft(X/S).real[:orig_N]
    return normalise(y, power)


def violet(N, power):
    orig_N = N
    # Because doing rfft->ifft produces different length outputs depending if its odd or even length inputs
    N+=1
    x = np.random.randn(N).astype(np.float32)
    X = rfft(x) / N
    S = np.arange(X.size)  # Filter
    y = irfft(X*S).real[0:orig_N]
    return normalise(y, power)


def generate_colored_gaussian_noise(file_path='./sample_audio.wav', snr=10, color='white'):

    # Load audio data into a 1D numpy array
    un_noised_file, _ = torchaudio.load(file_path)
    un_noised_file = un_noised_file.numpy()
    un_noised_file = np.reshape(un_noised_file, -1)

    # Create an audio Power array
    un_noised_file_watts = un_noised_file ** 2

    # Create an audio Decibal array
    un_noised_file_db = 10 * np.log10(un_noised_file_watts)

    # Calculate signal power and convert to dB
    un_noised_file_avg_watts = np.mean(un_noised_file_watts)
    un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)

    # Calculate noise power
    added_noise_avg_db = un_noised_file_avg_db - snr
    added_noise_avg_watts = 10 ** (added_noise_avg_db / 10)

    # Generate a random sample of additive gaussian noise
    added_noise = noise(len(un_noised_file), color, added_noise_avg_watts)

    # Add Noise to the Un-Noised signal
    noised_audio = un_noised_file + added_noise

    return noised_audio


def load_audio_file(file_path='./sample_audio.wav'):
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.numpy()
    waveform = np.reshape(waveform, -1)
    return waveform

def save_audio_file(np_array=np.array([0.5]*1000),file_path='./sample_audio.wav', sample_rate=48000, bit_precision=16):
    np_array = np.reshape(np_array, (1,-1))
    torch_tensor = torch.from_numpy(np_array)
    torchaudio.save(file_path, torch_tensor, sample_rate, precision=bit_precision)
