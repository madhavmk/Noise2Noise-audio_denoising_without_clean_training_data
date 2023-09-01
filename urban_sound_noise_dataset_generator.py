import numpy as np
from scipy import interpolate
from scipy.io import wavfile
import os
import random

import warnings

warnings.filterwarnings("ignore")
np.random.seed(999)

noise_class_dictionary = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

import torchaudio


# Set Audio backend as Sounfile for windows and Sox for Linux
torchaudio.set_audio_backend("soundfile")

from pydub import AudioSegment


def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(
            0, duration, int(original.shape[0] * new_rate / old_rate)
        )
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


fold_names = []
for i in range(1, 11):
    fold_names.append("fold" + str(i) + "/")


def diffNoiseType(files, noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] != str(noise_type):
                result.append(i)
    return result


def oneNoiseType(files, noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] == str(noise_type):
                result.append(i)
    return result


def genNoise(filename, num_per_fold, dest):
    true_path = target_folder + "/" + filename
    audio_1 = AudioSegment.from_file(true_path)
    counter = 0
    for fold in fold_names:
        dirname = Urban8Kdir + fold
        dirlist = os.listdir(dirname)
        total_noise = len(dirlist)
        samples = np.random.choice(total_noise, num_per_fold, replace=False)
        for s in samples:
            noisefile = dirlist[s]
            try:
                audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
                combined = audio_1.overlay(audio_2, times=5)
                target_dest = (
                    dest
                    + "/"
                    + filename[: len(filename) - 4]
                    + "_noise_"
                    + str(counter)
                    + ".wav"
                )
                combined.export(target_dest, format="wav")
                counter += 1
            except:
                print("Some kind of audio decoding error occurred, skipping this case")


def makeCorruptedFile_singletype(filename, dest, noise_type, snr):
    succ = False
    true_path = target_folder + "/" + filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except:
            print(
                "Some kind of audio decoding error occurred for base file... skipping"
            )
            break

        un_noised_file, _ = torchaudio.load(true_path)
        un_noised_file = un_noised_file.numpy()
        un_noised_file = np.reshape(un_noised_file, -1)
        # Create an audio Power array
        un_noised_file_watts = un_noised_file**2
        # Create an audio Decibal array
        un_noised_file_db = 10 * np.log10(un_noised_file_watts)
        # Calculate signal power and convert to dB
        un_noised_file_avg_watts = np.mean(un_noised_file_watts)
        un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
        # Calculate noise power
        added_noise_avg_db = un_noised_file_avg_db - snr
        try:
            fold = np.random.choice(fold_names, 1, replace=False)
            fold = fold[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = oneNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            samples = np.random.choice(total_noise, 1, replace=False)
            s = samples[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname + "/" + noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file**2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest + "/" + filename
            combined.export(target_dest, format="wav")
            succ = True
        except:
            pass
            # print("Some kind of audio decoding error occurred for the noise file..retrying")


def makeCorruptedFile_differenttype(filename, dest, noise_type, snr):
    succ = False
    true_path = target_folder + "/" + filename
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except:
            print(
                "Some kind of audio decoding error occurred for base file... skipping"
            )
            break

        un_noised_file, _ = torchaudio.load(true_path)
        un_noised_file = un_noised_file.numpy()
        un_noised_file = np.reshape(un_noised_file, -1)
        # Create an audio Power array
        un_noised_file_watts = un_noised_file**2
        # Create an audio Decibal array
        un_noised_file_db = 10 * np.log10(un_noised_file_watts)
        # Calculate signal power and convert to dB
        un_noised_file_avg_watts = np.mean(un_noised_file_watts)
        un_noised_file_avg_db = 10 * np.log10(un_noised_file_avg_watts)
        # Calculate noise power
        added_noise_avg_db = un_noised_file_avg_db - snr

        try:
            fold = np.random.choice(fold_names, 1, replace=False)
            fold = fold[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            possible_noises = diffNoiseType(dirlist, noise_type)
            total_noise = len(possible_noises)
            samples = np.random.choice(total_noise, 1, replace=False)
            s = samples[0]
            noisefile = possible_noises[s]

            noise_src_file, _ = torchaudio.load(dirname + "/" + noisefile)
            noise_src_file = noise_src_file.numpy()
            noise_src_file = np.reshape(noise_src_file, -1)
            noise_src_file_watts = noise_src_file**2
            noise_src_file_db = 10 * np.log10(noise_src_file_watts)
            noise_src_file_avg_watts = np.mean(noise_src_file_watts)
            noise_src_file_avg_db = 10 * np.log10(noise_src_file_avg_watts)

            db_change = added_noise_avg_db - noise_src_file_avg_db

            audio_2 = AudioSegment.from_file(dirname + "/" + noisefile)
            audio_2 = audio_2 + db_change
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest + "/" + filename
            combined.export(target_dest, format="wav")
            succ = True
        except:
            pass


Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/clean_trainset_28spk_wav"

for key in noise_class_dictionary:
    print("\t{} : {}".format(key, noise_class_dictionary[key]))

noise_type = int(input("Enter the noise class dataset to generate :\t"))

inp_folder = "Datasets/US_Class" + str(noise_type) + "_Train_Input"
op_folder = "Datasets/US_Class" + str(noise_type) + "_Train_Output"

print("Generating Training Data..")
print("Making train input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)
print("Making train output folder")
if not os.path.exists(op_folder):
    os.makedirs(op_folder)

from tqdm import tqdm

counter = 0
# noise_type = 1
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        # noise_type=random.randint(0,9)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        snr = random.randint(0, 10)
        makeCorruptedFile_differenttype(filename, op_folder, noise_type, snr)
        counter += 1


Urban8Kdir = "Datasets/UrbanSound8K/audio/"
target_folder = "Datasets/clean_testset_wav"
inp_folder = "Datasets/US_Class" + str(noise_type) + "_Test_Input"

print("Generating Testing Data..")
print("Making test input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)

counter = 0
# noise type was specified earlier
for file in tqdm(os.listdir(target_folder)):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        snr = random.randint(0, 10)
        makeCorruptedFile_singletype(filename, inp_folder, noise_type, snr)
        counter += 1
