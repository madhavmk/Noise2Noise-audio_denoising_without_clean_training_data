from metrics_utils import *
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
from scipy.linalg import solve_toeplitz,toeplitz
import pesq as pypesq
from pystoi import stoi
import random

# Expected input, 2 numpy arrays, one for the reference clean audio, the other for the degraded audio, and sampling rate (should be same)
# The way we'd use these metrics would be to compute the values on clean compared to noisy and then clean compared to our denoising results

class AudioMetrics():
    def __init__(self, target_speech, input_speech, fs): 
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")
    
        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # The SSNR and composite metrics fail when comparing silence
        # The minimum value of the signal is clipped to 0.001 or -0.001 to overcome that. For reference, in a non-silence case, the minimum value was around 40 (???? Find correct value)
        # For PESQ and STOI, results are identical regardless of wether or not 0 is present

        # The Metrics are as follows:
        # SSNR : Segmented Signal to noise ratio - Capped from [-10,35] (higher is better)
        # PESQ : Perceptable Estimation of Speech Quality - Capped from [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - From 0 to 1
        # CSIG : Quality of Speech Signal. Ranges from 1 to 5 (Higher is better)
        # CBAK : Quality of Background intrusiveness. Ranges from 1 to 5 (Higher is better - less intrusive)
        # COVL : Overall Quality measure. Ranges from 1 to 5 (Higher is better)
        # CSIG,CBAK and COVL are computed using PESQ and some other metrics like LLR and WSS
        
        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # If value less than min_cutoff difference from 0, then clip
            '''
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    clean_speech[index] = self.clip_values[0]
                else:
                    clean_speech[index] = self.clip_values[1]
            '''
            if data==0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data
                
        for index, data in np.ndenumerate(input_speech):
            '''
            # If value less than min_cutoff difference from 0, then clip
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    processed_speech[index] = self.clip_values[0]
                else:
                    processed_speech[index] = self.clip_values[1]
            '''
            if data==0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data
             
                
        #print('clean speech: ', clean_speech)
        #print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech,fs)
        self.PESQ = pesq_score(clean_speech, processed_speech, fs, force_resample=True)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)
        self.CSIG, self.CBAK, self.COVL = composite(clean_speech, processed_speech, fs)

    def display(self):
        fstring = "{} : {:.3f}"
        metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR"]
        for name in metric_names:
            metric_value = eval("self."+name)
            print(fstring.format(name,metric_value))
        
        
class AudioMetrics2():
    def __init__(self, target_speech, input_speech, fs): 
        if len(target_speech) != len(input_speech):
            raise AudioMetricsException("Signal lengths don't match!")
    
        self.min_cutoff = 0.01
        self.clip_values = (-self.min_cutoff, self.min_cutoff)

        # The SSNR and composite metrics fail when comparing silence
        # The minimum value of the signal is clipped to 0.001 or -0.001 to overcome that. For reference, in a non-silence case, the minimum value was around 40 (???? Find correct value)
        # For PESQ and STOI, results are identical regardless of wether or not 0 is present

        # The Metrics are as follows:
        # SSNR : Segmented Signal to noise ratio - Capped from [-10,35] (higher is better)
        # PESQ : Perceptable Estimation of Speech Quality - Capped from [-0.5, 4.5]
        # STOI : Short Term Objective Intelligibilty of Speech - From 0 to 1
        # CSIG : Quality of Speech Signal. Ranges from 1 to 5 (Higher is better)
        # CBAK : Quality of Background intrusiveness. Ranges from 1 to 5 (Higher is better - less intrusive)
        # COVL : Overall Quality measure. Ranges from 1 to 5 (Higher is better)
        # CSIG,CBAK and COVL are computed using PESQ and some other metrics like LLR and WSS
        
        clean_speech = np.zeros(shape=target_speech.shape)
        processed_speech = np.zeros(shape=input_speech.shape)

        for index, data in np.ndenumerate(target_speech):
            # If value less than min_cutoff difference from 0, then clip
            '''
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    clean_speech[index] = self.clip_values[0]
                else:
                    clean_speech[index] = self.clip_values[1]
            '''
            if data==0:
                clean_speech[index] = 0.01
            else:
                clean_speech[index] = data
                
        for index, data in np.ndenumerate(input_speech):
            '''
            # If value less than min_cutoff difference from 0, then clip
            if data<=self.min_cutoff and data>=-self.min_cutoff:
                if data < 0:
                    processed_speech[index] = self.clip_values[0]
                else:
                    processed_speech[index] = self.clip_values[1]
            '''
            if data==0:
                processed_speech[index] = 0.01
            else:
                processed_speech[index] = data
             
                
        #print('clean speech: ', clean_speech)
        #print('processed speech : ', processed_speech)
        self.SNR = snr(target_speech, input_speech)
        self.SSNR = SNRseg(target_speech, input_speech,fs)
        self.STOI = stoi_score(clean_speech, processed_speech, fs)

# Formula Reference: http://www.irisa.fr/armor/lesmembres/Mohamed/Thesis/node94.html

def snr(reference, test):
    numerator = 0.0
    denominator = 0.0
    for i in range(len(reference)):
        numerator += reference[i]**2
        denominator += (reference[i] - test[i])**2
    return 10*np.log10(numerator/denominator)


# Reference : https://github.com/schmiph2/pysepm

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)



def composite(clean_speech, processed_speech, fs):
    wss_dist=wss(clean_speech, processed_speech, fs)
    llr_mean=llr(clean_speech, processed_speech, fs,used_for_composite=True)
    segSNR=SNRseg(clean_speech, processed_speech, fs)
    pesq_mos,mos_lqo = pesq(clean_speech, processed_speech,fs)    
    if fs >= 16e3:
        used_pesq_val = mos_lqo
    else:
        used_pesq_val = pesq_mos    

    Csig = 3.093 - 1.029*llr_mean + 0.603*used_pesq_val-0.009*wss_dist
    Csig = np.max((1,Csig))  
    Csig = np.min((5, Csig)) # limit values to [1, 5]
    Cbak = 1.634 + 0.478 *used_pesq_val - 0.007*wss_dist + 0.063*segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5,Cbak)) # limit values to [1, 5]
    Covl = 1.594 + 0.805*used_pesq_val - 0.512*llr_mean - 0.007*wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl)) # limit values to [1, 5]
    return Csig,Cbak,Covl

def pesq_score(clean_speech, processed_speech, fs, force_resample=False):
    if fs!=8000 or fs!=16000:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 16000)
            processed_speech = resample(processed_speech, fs, 16000)
            fs = 16000
        else:
            raise(AudioMetricsException("Invalid sampling rate for PESQ! Need 8000 or 16000Hz but got "+str(fs)+"Hz"))
    if fs==16000:
        score = pypesq.pesq(16000, clean_speech, processed_speech, 'wb')
        score = min(score,4.5)
        score = max(-0.5,score)
        return(score)
    else:
        score = pypesq.pesq(16000, clean_speech, processed_speech, 'nb')
        score = min(score,4.5)
        score = max(-0.5,score)
        return(score)

# Original paper http://cas.et.tudelft.nl/pubs/Taal2010.pdf
# Says to resample to 10kHz if not already at that frequency. I've kept options to adjust
def stoi_score(clean_speech, processed_speech, fs, force_resample=True, force_10k=True):
    if fs!=10000 and force_10k==True:
        if force_resample:
            clean_speech = resample(clean_speech, fs, 10000)
            processed_speech = resample(processed_speech, fs, 10000)
            fs = 10000
        else:
            raise(AudioMetricsException("Forced 10kHz sample rate for STOI. Got "+str(fs)+"Hz"))
    return stoi(clean_speech, processed_speech, 10000, extended=False)
