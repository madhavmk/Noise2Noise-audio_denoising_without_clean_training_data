import wav_noiser as noiser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Directory containing Un-Noised files
UN_NOISED_DIRECTORY = Path('../Datasets/clean_trainset_28spk_wav')
#UN_NOISED_DIRECTORY = Path('../clean_testset_wav')
un_noised_directory_wav_files = sorted(list(UN_NOISED_DIRECTORY.rglob('*.wav')))
print("Total clean samples:",len(un_noised_directory_wav_files))

# Directory where you want to save the Noised files
#if not os.path.exists('../Datasets/white_train_input'):
#    os.makedirs('../Datasets/white_train_input')

if not os.path.exists('../Datasets/white_train_target'):
    os.makedirs('../Datasets/white_train_target')

#NOISED_DIRECTORY_INPUT = Path('../Datasets/white_train_input')
NOISED_DIRECTORY_OUTPUT = Path('../Datasets/white_train_target')

ct = 0

for audio_file in tqdm(un_noised_directory_wav_files):
	ct +=1
	#if ct %100 == 0:
	#	print(ct)
	un_noised_file = noiser.load_audio_file(file_path=audio_file)
	#random_snr = np.random.randint(0,10)
	#white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
	#noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(str(NOISED_DIRECTORY_INPUT),
	#				audio_file.name[:len(audio_file.name)-4] + "_inp.wav"))
	random_snr = np.random.randint(0,10)
	white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
	noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(str(NOISED_DIRECTORY_OUTPUT),
					audio_file.name))

print()
print("Training data generated")
print()
'''
UN_NOISED_DIRECTORY = Path('../Datasets/clean_testset_wav')
un_noised_directory_wav_files = sorted(list(UN_NOISED_DIRECTORY.rglob('*.wav')))
print("Total clean samples:",len(un_noised_directory_wav_files))

ct = 0
if not os.path.exists('../Datasets/noisy_testset_white'):
    os.makedirs('../Datasets/noisy_testset_white')
NOISED_DIRECTORY= Path('../Datasets/noisy_testset_white')

for audio_file in un_noised_directory_wav_files:
	ct +=1
	if ct %100 == 0:
		print(ct)
	un_noised_file = noiser.load_audio_file(file_path=audio_file)
	random_snr = np.random.randint(0,10)
	white_gaussian_noised_audio = noiser.gen_colored_gaussian_noise(file_path=audio_file, snr=random_snr, color='white')
	noiser.save_audio_file(np_array=white_gaussian_noised_audio, file_path='{}/{}'.format(str(NOISED_DIRECTORY),
					audio_file.name[:len(audio_file.name)-4] + "_test.wav"))

print()
print("Testing data generated")
print()
'''
