import colored_noise_utils as noiser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm


TRAINING_INPUT_PATH = 'Datasets/WhiteNoise_Train_Input'
TRAINING_OUTPUT_PATH = 'Datasets/WhiteNoise_Train_Output'
TESTING_INPUT_PATH = 'Datasets/WhiteNoise_Test_Input'


CLEAN_TRAINING_DIR = Path('Datasets/clean_trainset_28spk_wav')
CLEAN_TESTING_DIR = Path("Datasets/clean_testset_wav")
clean_training_dir_wav_files = sorted(list(CLEAN_TRAINING_DIR.rglob('*.wav')))
clean_testing_dir_wav_files = sorted(list(CLEAN_TESTING_DIR.rglob('*.wav')))
print("Total training samples:",len(clean_training_dir_wav_files))

print("Generating Training data")
if not os.path.exists(TRAINING_INPUT_PATH):
    os.makedirs(TRAINING_INPUT_PATH)
if not os.path.exists(TRAINING_OUTPUT_PATH):
    os.makedirs(TRAINING_OUTPUT_PATH)


for audio_file in tqdm(clean_training_dir_wav_files):
    un_noised_file = noiser.load_audio_file(file_path=audio_file)
    
    random_snr = np.random.randint(0,10)
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
    noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(TRAINING_INPUT_PATH,audio_file.name))
    
    random_snr = np.random.randint(0,10)
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
    noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(TRAINING_OUTPUT_PATH,audio_file.name))

    
print("Generating Testing data")
if not os.path.exists(TESTING_INPUT_PATH):
    os.makedirs(TESTING_INPUT_PATH)
    
for audio_file in tqdm(clean_testing_dir_wav_files):
    un_noised_file = noiser.load_audio_file(file_path=audio_file)

    random_snr = np.random.randint(0,10)
    white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
    noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(TESTING_INPUT_PATH,audio_file.name))
