# Noise2Noise-audio_denoising_without_clean_training_data
Source code for the paper titled "Speech Denoising without Clean Training Data: a Noise2Noise Approach". This paper tackles the problem of the heavy dependence of clean speech data required by deep learning based audio denoising methods by showing that it is possible to train deep speech denoising networks using only noisy speech samples.

## Data Generation
To generate the data used, please download the speech samples from [here](https://datashare.ed.ac.uk/handle/10283/2791), and the Urbansound noise samples from [here](https://urbansounddataset.weebly.com/urbansound8k.html)

Create a new folder named Datasets in the parent directory of the repository. After extracting the audio into the datasets folder, run TargetUSGen.py to generate the data for the Urbansound distributions. The allowed noise types are 0 to 9.
To create white noise please use batch_noiser.py , and to generate the mixed category, modify TargetUSGen to use a random noise type instead of an input.

## Running the notebook
Update Cell 1 with the appropriate noise type and training category (either N2N or Baseline). Directories to store the results will be generated automatically. 
