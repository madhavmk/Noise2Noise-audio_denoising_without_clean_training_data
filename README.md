# Speech Denoising without Clean Training Data: a Noise2Noise Approach
Source code for the paper titled "Speech Denoising without Clean Training Data: a Noise2Noise Approach". This paper removes the obstacle of the heavy dependence of clean speech data required by deep learning based audio denoising methods by showing that it is possible to train deep speech denoising networks using only noisy speech samples. Furthermore it is revealed that training regimes using only noisy audio targets achieve superior denoising performance over conventional training regimes utilizing clean training audio targets, in cases involving complex noise distributions and low Signal-to-Noise ratios (high noise environments). This is demonstrated through experiments studying the efficacy of our proposed approach over both real-world noises and synthetic noises using the 20 layered Deep Complex U-Net architecture. We aim to incentivise the collection of audio
data, even when the circumstances are not ideal to allow it to be perfectly clean. We believe that this could significantly advance the prospects of speech denoising technologies for various lowresource languages, due to the decreased costs and barriers in data collection.

# Research Paper and Citation
The research paper is put up on Arxiv. See the link : https://arxiv.org/abs/2104.03838. We have submitted the work to INTERSPEECH 2021(under review).

If you would like to cite this work, please use the following Bibtex citation:

@misc{kashyap2021speech,
      title={Speech Denoising without Clean Training Data: a Noise2Noise Approach}, 
      author={Madhav Mahesh Kashyap and Anuj Tambwekar and Krishnamoorthy Manohara and S Natarajan},
      year={2021},
      eprint={2104.03838},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}

## Data Generation
To generate the data used, please download the speech samples from [here](https://datashare.ed.ac.uk/handle/10283/2791), and the Urbansound noise samples from [here](https://urbansounddataset.weebly.com/urbansound8k.html)

Create a new folder named Datasets in the parent directory of the repository. After extracting the audio into the datasets folder, run TargetUSGen.py to generate the data for the Urbansound distributions. The allowed noise types are 0 to 9.
To create white noise please use batch_noiser.py , and to generate the mixed category, modify TargetUSGen to use a random noise type instead of an input.

## Running the notebook
Update Cell 1 with the appropriate noise type and training category (either N2N or Baseline). Directories to store the results will be generated automatically. 
