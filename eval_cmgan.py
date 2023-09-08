#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

See the description of the functions 'run_baseline' and 'compute_metrics'. 

The function 'compute_metrics' is independent of the baseline system, you
may use it to report the results of your system for the UDASE task of CHiME-7.

Example usage
-------------

# You must first set appropriately the following variables below: 
# 'output_path', 'chime5_input_path', 'reverberant_librichime5_input_path',
# 'librimix_input_path'.

# run the 'remixit-vad' baseline on CHiME-5, reverberant LibriCHiME-5, Librimix:
python eval.py --run-baseline --model remixit-vad

# evaluate the 'remixit-vad' baseline results on CHiME-5, 
# reverberant LibriCHiME-5, and Librimix, and compute the performance scores
# on the unprocessed input signals.
python eval.py --eval-baseline --input-scores --model remixit-vad

"""

# output path to store the results of the baseline
#output_path = '/data2/datasets/UDASE-CHiME2023/baseline_results'
output_path = '/mnt/parscratch/users/acp20glc/UDASE-CHiME2023_new/CMGAN_dnsmos_sig_1.0'
######## Config for the CHiME-5 dataset of the UDASE task ########

# path to the input data    
chime5_input_path = '/mnt/parscratch/users/acp20glc/CHiME2023/data/CHiME-5/CHiME5_processed/'
# subsets to process
chime5_subsets_run = ['eval/1', 'eval/listening_test']
#chime5_subsets_run = []
# subsets to evaluate
chime5_subsets_eval = ['eval/1']
#chime5_subsets_eval = []
######## Config for the Reverberant LibriCHiME-5 dataset of the UDASE task ########

# path to the input data
reverberant_librichime5_input_path = '/mnt/parscratch/users/acp20glc/CHiME2023/data/reverberant-LibriCHiME-5/'
# subsets to process and evaluate
reverberant_librichime5_subsets = ['eval/1', 'eval/2', 'eval/3']
#reverberant_librichime5_subsets = []
######## Config for the LibriMix dataset ########

# path to the input data
librimix_input_path = '/mnt/parscratch/users/acp20glc/CHiME2023/data/LibriMix/'
# subsets to process and evaluate
#librimix_subsets = ['Libri2Mix/wav16k/max/test/mix_single', 
#                    'Libri2Mix/wav16k/max/test/mix_both',
#                    'Libri3Mix/wav16k/max/test/mix_both']
librimix_subsets = []

#%% imports

import numpy as np
import torch
import torchaudio
import metrics.dnnmos_metric as dnnmos_metric
import metrics.sisdr_metric as sisdr_metric
import utils.mixture_consistency as mixture_consistency
from models.CMGAN.generator import TSCNet
import pandas as pd
import os
import glob
import soundfile as sf
from tqdm import tqdm
import pyloudnorm as pyln
import argparse
import warnings
import torch.nn.functional as F
from pysepm import composite
from speechbrain.processing.features import STFT,ISTFT
stft_func = STFT(sample_rate=16000,
                win_length=25,
                hop_length=6.25,
                n_fft=400,
                window_fn=torch.hamming_window,
                onesided=True
                )
istft_func = ISTFT(sample_rate=16000,
                    win_length=25,
                    hop_length=6.25,
                    window_fn=torch.hamming_window
                    )
warnings.filterwarnings("ignore")

#%%

def normalize(x, target_loudness=-30, meter=None, sr=16000):
    """
    LUFS normalization of a signal using pyloudnorm.
    
    Parameters
    ----------
    x : ndarray
        Input signal.
    target_loudness : float, optional
        Target loudness of the output in dB LUFS. The default is -30.
    meter : Meter, optional
        The pyloudnorm BS.1770 meter. The default is None.
    sr : int, optional
        Sampling rate. The default is 16000.

    Returns
    -------
    x_norm : ndarray
        Normalized output signal.
    """
    
    if meter is None:
        meter = pyln.Meter(sr) # create BS.1770 meter
    
    # peak normalize to 0.7 to ensure that the meter does not return -inf
    x = x - np.mean(x)
    x = x/(np.max(np.abs(x)) + 1e-9)*0.7
            
    # measure the loudness first 
    loudness = meter.integrated_loudness(x)
    
    # loudness normalize audio to target_loudness LUFS
    x_norm = pyln.normalize.loudness(x, loudness, target_loudness)  

    return x_norm


def run_baseline(checkpoint, 
                 output_path, 
                 datasets=['chime-5', 'reverberant-librichime-5', 'librimix'],
                 chime5_input_path=None, 
                 chime5_subsets=['eval/1'], 
                 reverberant_librichime5_input_path=None, 
                 reverberant_librichime5_subsets=['eval/1', 'eval/2', 'eval/3'],
                 librimix_input_path=None,
                 librimix_subsets=['Libri2Mix/wav16k/max/test/mix_single', 
                                   'Libri2Mix/wav16k/max/test/mix_both',
                                   'Libri3Mix/wav16k/max/test/mix_both'],
                 save_mix=False, 
                 save_noise=False):
    
    """
    Run the baseline model of the CHiME-7 UDASE task.
    
    Parameters
    ----------
    checkpoint : string
        Path to the baseline model checkpoint.
    output_path : string
        Path to save the output signals.
    datasets : list of string, optional
        The list of the datasets to process. Each element of this list should be
        either 'chime-5', 'reverberant-librichime-5', or 'librimix'.        
        If a dataset is in the list (e.g., 'chime-5'), the other input 
        variables associated with this dataset should be appropriately set 
        (e.g., 'chime5_input_path' and 'chime5_subsets').
    chime5_input_path : string, optional
        Path to the (preprocessed) CHiME-5 dataset (as provided for the UDASE task).
        Default: None
    chime5_subsets : list of string, optional
        Subsets of the CHiME-5 dataset to process.
        Default: ['eval/1']
    reverberant_librichime5_input_path : string, optional
        Path to the reverberant LibriCHiME-5 dataset.
        Default: None
    reverberant_librichime5_subsets : list of string, optional
        Subsets of the reverberant LibriCHiME-5 dataset to process.
        Default: ['eval/1', 'eval/2', 'eval/3']
    librimix_input_path : string, optional
        Path to the reverberant LibriMix dataset.
        Default: None
    librimix_subsets : list of string, optional
        Subsets of theLibriMix dataset to process.
        Default: ['Libri2Mix/wav16k/max/test/mix_single', 
                 'Libri2Mix/wav16k/max/test/mix_both',
                 'Libri3Mix/wav16k/max/test/mix_both']
    save_mix : boolean, optional
        Boolean indicating if the input noisy mixture signal should be saved. 
        Default: False.
    save_noise : boolean, optional
        Boolean indicating if the estimated noise signal should be saved. 
        Default: False.
    
    Returns
    -------
    None.
    """
    output_path = os.path.join(output_path, 'audio')
            
    sr = 16000
    meter = pyln.Meter(sr)
    
    chime5_output_path = os.path.join(output_path, 'CHiME-5')
    reverberant_librichime5_output_path = os.path.join(output_path, 
                                                       'reverberant-LibriCHiME-5')
    librimix_output_path = os.path.join(output_path, 'LibriMix')    
    
    # model

    model = load_cmgan_model(checkpoint)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # process datasets   

    for dataset in datasets:
        
        if dataset == 'chime-5':
            print('Running the baseline on CHiME-5')
            subsets = chime5_subsets
        elif dataset == 'reverberant-librichime-5':
            print('Running the baseline on Reverberant LibriCHiME-5')
            subsets = reverberant_librichime5_subsets
        elif dataset == 'librimix':
            print('Running the baseline on LibriMix')
            subsets = librimix_subsets
            
        for subset in subsets:

            if dataset == 'chime-5':
                pattern = os.path.join(chime5_input_path, subset, '*.wav')
                curr_output_dir = os.path.join(chime5_output_path, subset)
            elif dataset == 'reverberant-librichime-5':
                pattern = os.path.join(reverberant_librichime5_input_path, subset,'*mix.wav')
                curr_output_dir = os.path.join(reverberant_librichime5_output_path, subset)
            elif dataset == 'librimix':
                pattern = os.path.join(librimix_input_path, subset,'*.wav')
                curr_output_dir = os.path.join(librimix_output_path, subset)
            
            if not os.path.isdir(curr_output_dir):
                os.makedirs(curr_output_dir)
            
            file_list = glob.glob(pattern)
            
            for ind, mix_file in tqdm(enumerate(file_list), total=len(file_list)):
                            
                file_name = os.path.basename(mix_file)
                file_name = os.path.splitext(file_name)[0]
                
                # Load the mixture
                mixture, _ = torchaudio.load(mix_file) # audio file should be mono channel
                
                # Pad the mixture
                file_length = mixture.shape[-1]
                min_k = int(np.ceil(np.log2(file_length/16000)))
                padded_length = 2**max(min_k, 1) * 16000
                input_mix = torch.zeros((1, padded_length), dtype=mixture.dtype)
                input_mix[..., :file_length] = mixture
                
                # Scale the mixture
                input_mix = input_mix.unsqueeze(1).cuda() 
                input_mix_std = input_mix.std(-1, keepdim=True)
                input_mix_mean = input_mix.mean(-1, keepdim=True)
                input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)
                
                # Perform inference and apply mixture consistency
                with torch.no_grad():
                    estimates = model(input_mix)
                    estimates = mixture_consistency.apply(estimates, input_mix)
            
                # Cut the mixture and estimates to original length (before padding)
                speech_est = estimates[0, 0, :file_length].cpu().numpy().squeeze()
                if save_noise:
                    noise_est = estimates[0, 1, :file_length].cpu().numpy().squeeze()
                if save_mix:
                    input_mix = input_mix[0, 0, :file_length].cpu().numpy().squeeze()
            
                # Normalize to -30 LUFS. This is only mandatory for the CHiME-5 
                # dataset (to compute DNS-MOS and for listening tests), but it
                # should not affect the SI-SDR results on other datasets as
                # the SI-SDR is scale invariant.
                speech_est = normalize(speech_est, target_loudness=-30, meter=meter, sr=16000)
                if save_noise:
                    noise_est = normalize(noise_est, target_loudness=-30, meter=meter, sr=16000)
                if save_mix:    
                    input_mix = normalize(input_mix, target_loudness=-30, meter=meter, sr=16000)
                
                # Save
                if dataset == 'chime-5' or dataset == 'librimix':
                    mix_id = file_name
                elif dataset == 'reverberant-librichime-5':
                    mix_id = file_name[:-4]
                
                sf.write(os.path.join(curr_output_dir, mix_id + '_output.wav'), speech_est, sr)
                
                if save_noise:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_noise_estimate.wav'), noise_est, sr)
                if save_mix:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_mix.wav'), input_mix, sr)    


def load_cmgan_model(path):
    model = TSCNet()
    model.load_state_dict(torch.load(path))
    print(f"Fetched model from: {path}")

    return model


def compute_feats_real_imag(wavs):
        """Feature computation pipeline"""
        #print("WAVS",wavs.shape)
        feats =compute_STFT(wavs)
        #print("FEATS",feats.shape) # B * T * F * 2 
        feats_spec = power_compress(feats) # B * 2 * T * F
    
        return feats_spec

def compute_STFT(wavs):
 
    return stft_func(wavs.squeeze(1))

def compute_ISTFT(inputs):

    return istft_func(inputs)

def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1./0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)

def run_CMGAN(checkpoint, 
                 output_path, 
                 datasets=['chime-5', 'reverberant-librichime-5', 'librimix'],
                 chime5_input_path=None, 
                 chime5_subsets=['eval/1'], 
                 reverberant_librichime5_input_path=None, 
                 reverberant_librichime5_subsets=['eval/1', 'eval/2', 'eval/3'],
                 librimix_input_path=None,
                 librimix_subsets=['Libri2Mix/wav16k/max/test/mix_single', 
                                   'Libri2Mix/wav16k/max/test/mix_both',
                                   'Libri3Mix/wav16k/max/test/mix_both'],
                 save_mix=False, 
                 save_noise=False):
    
    """
    Run the baseline model of the CHiME-7 UDASE task.
    
    Parameters
    ----------
    checkpoint : string
        Path to the baseline model checkpoint.
    output_path : string
        Path to save the output signals.
    datasets : list of string, optional
        The list of the datasets to process. Each element of this list should be
        either 'chime-5', 'reverberant-librichime-5', or 'librimix'.        
        If a dataset is in the list (e.g., 'chime-5'), the other input 
        variables associated with this dataset should be appropriately set 
        (e.g., 'chime5_input_path' and 'chime5_subsets').
    chime5_input_path : string, optional
        Path to the (preprocessed) CHiME-5 dataset (as provided for the UDASE task).
        Default: None
    chime5_subsets : list of string, optional
        Subsets of the CHiME-5 dataset to process.
        Default: ['eval/1']
    reverberant_librichime5_input_path : string, optional
        Path to the reverberant LibriCHiME-5 dataset.
        Default: None
    reverberant_librichime5_subsets : list of string, optional
        Subsets of the reverberant LibriCHiME-5 dataset to process.
        Default: ['eval/1', 'eval/2', 'eval/3']
    librimix_input_path : string, optional
        Path to the reverberant LibriMix dataset.
        Default: None
    librimix_subsets : list of string, optional
        Subsets of theLibriMix dataset to process.
        Default: ['Libri2Mix/wav16k/max/test/mix_single', 
                 'Libri2Mix/wav16k/max/test/mix_both',
                 'Libri3Mix/wav16k/max/test/mix_both']
    save_mix : boolean, optional
        Boolean indicating if the input noisy mixture signal should be saved. 
        Default: False.
    save_noise : boolean, optional
        Boolean indicating if the estimated noise signal should be saved. 
        Default: False.
    
    Returns
    -------
    None.
    """
    
    output_path = os.path.join(output_path, 'audio')
            
    sr = 16000
    meter = pyln.Meter(sr)
    
    chime5_output_path = os.path.join(output_path, 'CHiME-5')
    reverberant_librichime5_output_path = os.path.join(output_path, 
                                                       'reverberant-LibriCHiME-5')
    librimix_output_path = os.path.join(output_path, 'LibriMix')    
    
    # model

    model = load_cmgan_model(checkpoint)
    #model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # process datasets   

    for dataset in datasets:
        
        if dataset == 'chime-5':
            print('Running the baseline on CHiME-5')
            subsets = chime5_subsets
        elif dataset == 'reverberant-librichime-5':
            print('Running the baseline on Reverberant LibriCHiME-5')
            subsets = reverberant_librichime5_subsets
        elif dataset == 'librimix':
            print('Running the baseline on LibriMix')
            subsets = librimix_subsets
        #print(reverberant_librichime5_input_path)
        #print(subsets)
        
        for subset in subsets:

            if dataset == 'chime-5':
                pattern = os.path.join(chime5_input_path, subset, '*.wav')
                curr_output_dir = os.path.join(chime5_output_path, subset)
            elif dataset == 'reverberant-librichime-5':
                pattern = os.path.join(reverberant_librichime5_input_path, subset,'*_mix.wav')
                curr_output_dir = os.path.join(reverberant_librichime5_output_path, subset)
            elif dataset == 'librimix':
                pattern = os.path.join(librimix_input_path, subset,'*.wav')
                curr_output_dir = os.path.join(librimix_output_path, subset)
            
            if not os.path.isdir(curr_output_dir):
                os.makedirs(curr_output_dir)
            #print(pattern)
            #input(">>>")
            file_list = glob.glob(pattern)
            #print(file_list)
            for ind, mix_file in tqdm(enumerate(file_list), total=len(file_list)):
                            
                file_name = os.path.basename(mix_file)
                file_name = os.path.splitext(file_name)[0]
                
                # Load the mixture
                mixture, _ = torchaudio.load(mix_file) # audio file should be mono channel
                
                # Pad the mixture
                file_length = mixture.shape[-1]
                min_k = int(np.ceil(np.log2(file_length/16000)))
                padded_length = 2**max(min_k, 1) * 16000
                input_mix = torch.zeros((1, padded_length), dtype=mixture.dtype)
                input_mix[..., :file_length] = mixture
                
                # Scale the mixture
                input_mix = input_mix.unsqueeze(1).cuda() 
                input_mix_std = input_mix.std(-1, keepdim=True)
                input_mix_mean = input_mix.mean(-1, keepdim=True)
                input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)
                
                # Perform inference and apply mixture consistency
                # with torch.no_grad():

                #     noisy_spec = compute_feats_real_imag(input_mix)
                #     print("NOISY SPEC",noisy_spec.shape)
                #     #print(noisy_spec[:,0,:,:])
                #     predict_real,predict_imag = model(noisy_spec)
                #     print("MODEL OUT:")
                #     print(predict_real.shape,predict_imag.shape)
                #     predict_spec_uncompress = power_uncompress(predict_real, predict_imag).squeeze(1)

                #     estimates = compute_ISTFT(predict_spec_uncompress)
                #     print(estimates.shape)
                #     estimates = mixture_consistency.apply(estimates, input_mix)
                input_mix = input_mix.squeeze(1)
                with torch.no_grad():
                    css_len = 16000*4
                    shift = css_len // 2
                    #print("INPUT_MIX",input_mix.shape)
                    n_chunks = int(np.ceil(input_mix.shape[-1] / (shift))) -1
                    #print(n_chunks)
                    estimates = torch.zeros_like(input_mix).cuda()
                    #print("N_CHUNKS",n_chunks)
                    for i in range(n_chunks):
                        mix_chunk = input_mix[:, i * shift : (i + 1) * shift + shift]
                        if mix_chunk.shape[-1] < css_len:
                            pad = css_len - mix_chunk.shape[-1]
                            mix_chunk = F.pad(mix_chunk, (0, pad))
                        
                        #print("MIX CHUNK",mix_chunk.shape)
                        noisy_spec = compute_feats_real_imag(mix_chunk.unsqueeze(1))
                        #print("NOISY SPEC",noisy_spec.shape)
                        predict_real,predict_imag = model(noisy_spec.cuda())
                        #print("PREDICT REAL",predict_real.shape)
                        #print("PREDICT IMAG",predict_imag.shape)
                        predict_spec_uncompress = power_uncompress(predict_real, predict_imag).squeeze(1)
                        #print("PREDICT SPEC UNCOMPRESS",predict_spec_uncompress.shape)
                        est_source_chunk = compute_ISTFT(predict_spec_uncompress)
                        #print("EST SOURCE CHUNK",est_source_chunk.shape)
                        window = torch.hann_window(shift*2, device=est_source_chunk.device)
                        #print("WINDOW",window.shape)
                        if i == 0:
                            window[:shift] = 1.0
                        #elif i == n_chunks-1:
                        #    window[shift:] = 1.0
                        windowed_est_source_chunk = est_source_chunk * window
                        #print("WINDOWED EST SOURCE CHUNK",windowed_est_source_chunk.shape)
                        #print(i * shift, (i + 1) * shift+shift) 
                        
                        if i < n_chunks-1:
                            estimates[:, i * shift : (i + 1) * shift+shift] += windowed_est_source_chunk
                        else:
                            estimates[:, i * shift : input_mix.shape[-1]] += windowed_est_source_chunk[:, :input_mix.shape[-1] - i * shift]

                        #print(i * shift, input_mix.shape[-1])
                        #print(input_mix.shape[-1] - i * shift)
                estimates = estimates.unsqueeze(1)
                #print("ESTIMATES",estimates.shape)
                #input(">>>")
                # Cut the mixture and estimates to original length (before padding)
                speech_est = estimates[0, 0, :file_length].cpu().numpy().squeeze()
                if save_noise:
                    noise_est = estimates[0, 1, :file_length].cpu().numpy().squeeze()
                if save_mix:
                    input_mix = input_mix[0, 0, :file_length].cpu().numpy().squeeze()
            
                # Normalize to -30 LUFS. This is only mandatory for the CHiME-5 
                # dataset (to compute DNS-MOS and for listening tests), but it
                # should not affect the SI-SDR results on other datasets as
                # the SI-SDR is scale invariant.
                speech_est = normalize(speech_est, target_loudness=-30, meter=meter, sr=16000)
                if save_noise:
                    noise_est = normalize(noise_est, target_loudness=-30, meter=meter, sr=16000)
                if save_mix:    
                    input_mix = normalize(input_mix, target_loudness=-30, meter=meter, sr=16000)
                
                # Save
                if dataset == 'chime-5' or dataset == 'librimix':
                    mix_id = file_name
                elif dataset == 'reverberant-librichime-5':
                    mix_id = file_name[:-4]
                
                sf.write(os.path.join(curr_output_dir, mix_id + '_output.wav'), speech_est, sr)
                
                if save_noise:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_noise_estimate.wav'), noise_est, sr)
                if save_mix:
                    sf.write(os.path.join(curr_output_dir, mix_id + '_mix.wav'), input_mix, sr)    


def compute_metrics(output_path,
                    chime5_input_path=None, 
                    chime5_subsets=None, 
                    reverberant_librichime5_input_path=None, 
                    reverberant_librichime5_subsets=None,
                    librimix_input_path=None,
                    librimix_subsets=None,
                    compute_input_scores=False):
    """
    Script to compute the objective performance metrics for the UDASE task of 
    the CHiME-7 challenge.
        
    See Example Usage below for detailed information.

    Parameters
    ----------
    
    output_path : string
        Path where the results (output signals) are saved. 
        See Example usage for more details.
    chime5_input_path : string, optional
        Path to the (preprocessed) CHiME-5 dataset (as provided for the UDASE task).
    chime5_subsets : list of string, optional
        Subsets of the CHiME-5 dataset to process.
    reverberant_librichime5_input_path : string, optional
        Path to the reverberant LibriCHiME-5 dataset.
    reverberant_librichime5_subsets : list of string, optional
        Subsets of the reverberant LibriCHiME-5 dataset to process.
    librimix_input_path : string, optional
        Path to the reverberant LibriMix dataset.
    librimix_subsets : list of string, optional
        Subsets of theLibriMix dataset to process.
    compute_input_scores : boolean, optional
        Boolean indicating if the metrics should also be computed on the 
        input unprocessed noisy speech signal. The default is False.

    Returns
    -------
    None.
    
    Example usage
    -------------
    
    Assume the directory <output_path>/audio is structured as shown in the 
    tree below.

    <output_path>/audio
            │ 
            ├── CHiME-5
            │   └── eval
            │       └── 1
            ├── LibriMix
            │   ├── Libri2Mix
            │   │   └── wav16k
            │   │       └── max
            │   │           └── test
            │   │               ├── mix_both
            │   │               └── mix_single
            │   └── Libri3Mix
            │       └── wav16k
            │           └── max
            │               └── test
            │                   └── mix_both
            └── reverberant-LibriCHiME-5
                └── eval
                    ├── 1
                    ├── 2
                    └── 3
                
    At each leaf of this tree, we have a directory that contains the output
    wav files of the system. 
    
    The mandatory naming convention for the wav files is the following:
    - For CHiME-5, the output signal corresponding to the input signal
    <mix ID>.wav should be named <mix ID>_output.wav. For example, the 
    output signal <output_path>/audio/CHiME-5/eval/1/S01_P01_0_output.wav 
    corresponds to the input signal <chime5_input_path>/eval/1/S01_P01_0.wav
    - For reverberant LibriCHiME-5, the output signal corresponding to the input signal
    <mix ID>_mix.wav should be named <mix ID>_output.wav. For example, the 
    output signal <output_path>/audio/reverberant-LibriCHiME-5/eval/1/S01_P01_0a_output.wav
    corresponds to the input signal <reverberant_librichime5_input_path>/eval/1/S01_P01_0a_mix.wav
    - For LibriMix, the output signal corresponding to the input signal
    <mix ID>.wav should be named <mix ID>_output.wav. For example, the 
    output signal <output_path>/audio/LibriMix/Libri2Mix/wav16k/max/test/mix_single/61-70968-0000_8455-210777-0012_output.wav
    corresponds to the input signal <librimix_input_path>/Libri2Mix/wav16k/max/test/mix_single/61-70968-0000_8455-210777-0012.wav
        
    To compute the results for a given dataset, we should set the input 
    variables `*_input_path` and `*_subsets` appropriately (see Parameters 
    section above). If '*_input_path' and '*_subsets' are set to None 
    (default), results will not be computed for the corresponding dataset.
    
    For the example directory shown above, we set the input variables as 
    follows (this is just an example, of course you should adapt the paths):
        
        ##################
        ## CHiME-5 data ##
        ##################
        
        # path to the input data
        chime5_input_path = '/data2/datasets/UDASE-CHiME2023/CHiME-5'
        
        # subsets to process
        chime5_subsets = ['eval/1']
        
        ###################################
        ## Reverberant LibriCHiME-5 data ##
        ###################################
        
        # path to the input data
        reverberant_librichime5_input_path = '/data2/datasets/UDASE-CHiME2023/reverberant-LibriCHiME-5'
        
        # subsets to process
        reverberant_librichime5_subsets = ['eval/1', 'eval/2', 'eval/3']
        
        ###################
        ## LibriMix data ##
        ###################
        
        # path to the input data
        librimix_input_path = '/data/datasets/LibriMix'
        
        # subsets to process
        librimix_subsets = ['Libri2Mix/wav16k/max/test/mix_single', 
                            'Libri2Mix/wav16k/max/test/mix_both',
                            'Libri3Mix/wav16k/max/test/mix_both']
    
    After setting appropriately the variable 'output_path', for instance
        
        output_path = '/data2/datasets/UDASE-CHiME2023/baseline_results_new/remixit-vad'
        
    we can call the 'compute_metrics' function:
        
        compute_metrics(output_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets, 
                        reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                        reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                        librimix_input_path=librimix_input_path,
                        librimix_subsets=librimix_subsets)
    
    This function will save the objective performance results in a csv file 
    'results.csv' located in the folders CHiME-5, LibriMix or 
    reverberant-LibriCHiME-5 at <output_path>/csv.
    
    If for instance you only want to compute the results on the CHiME-5 dataset, then call:
    
        compute_metrics(output_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets, 
                        reverberant_librichime5_input_path=None, 
                        reverberant_librichime5_subsets=None,
                        librimix_input_path=None,
                        librimix_subsets=None)
    
    If the optional input variable 'compute_input_scores' is set to True, an
    additional csv file 'results_unprocessed.csv' will be saved at the same
    location as 'results.csv'. It will contain the performance scores for the
    unprocessed noisy speech signals.

    """
        
    if compute_input_scores:    
        meter = pyln.Meter(16000)
    
    # CHiME-5
    
    if chime5_input_path is not None and chime5_subsets is not None:
    
        print('Compute results on CHiME-5')
        
        chime5_output_path = os.path.join(output_path, 'audio', 'CHiME-5')
        chime5_df = pd.DataFrame(columns=['subset', 
                                          'input_file_name', 
                                          'output_file_name', 
                                          'SIG_MOS', 
                                          'BAK_MOS', 
                                          'OVR_MOS'])
        if compute_input_scores:
            unprocessed_chime5_df = pd.DataFrame(columns=['subset',
                                                          'input_file_name',
                                                          'output_file_name',
                                                          'SIG_MOS',
                                                          'BAK_MOS',
                                                          'OVR_MOS'])
        
        for subset in chime5_subsets:
            
            print(subset)
        
            input_file_list = glob.glob(os.path.join(chime5_input_path, subset,
                                                     '*.wav'))
                    
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
                
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                
                output_file = os.path.join(chime5_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
                
                speech_est, sr = sf.read(output_file)
                dnsmos_res = dnnmos_metric.compute_dnsmos(speech_est, fs=sr)
                    
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), dnsmos_res['sig_mos'],
                       dnsmos_res['bak_mos'], dnsmos_res['ovr_mos']]
                
                chime5_df.loc[len(chime5_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    mix = normalize(mix, target_loudness=-30, meter=meter, sr=16000)
                    
                    dnsmos_res = dnnmos_metric.compute_dnsmos(mix, fs=sr)
                        
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), dnsmos_res['sig_mos'],
                           dnsmos_res['bak_mos'], dnsmos_res['ovr_mos']]
                    
                    unprocessed_chime5_df.loc[len(unprocessed_chime5_df)] = row
        
        csv_dir = os.path.join(output_path, 'csv', 'CHiME-5')
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        csv_file = os.path.join(csv_dir, 'results.csv')
        
        chime5_df.to_csv(csv_file)
        
        if compute_input_scores:
            csv_file = os.path.join(output_path, 'csv', 'CHiME-5', 'results_unprocessed.csv')
            unprocessed_chime5_df.to_csv(csv_file)    
        
    # Reverberant LibriCHiME-5
    if reverberant_librichime5_input_path is not None and reverberant_librichime5_subsets is not None:
        
        print('Compute results on Reverberant LibriCHiME-5')
        
        reverberant_librichime5_output_path = os.path.join(output_path, 
                                                           'audio', 
                                                           'reverberant-LibriCHiME-5')
        
        reverberant_librichime5_df = pd.DataFrame(columns=['subset', 
                                                           'input_file_name', 
                                                           'output_file_name', 
                                                           'SI-SDR',
                                                           'comp_sig',
                                                           'comp_bak',
                                                           'comp_ovl'])
        if compute_input_scores:
            unprocessed_reverberant_librichime5_df = pd.DataFrame(columns=['subset',
                                                                           'input_file_name',
                                                                           'output_file_name',
                                                                           'SI-SDR'
                                                                           'comp_sig',
                                                                            'comp_bak',
                                                                            'comp_ovl'])
        
        for subset in reverberant_librichime5_subsets:
            
            print(subset)
            
            input_file_list = glob.glob(os.path.join(reverberant_librichime5_input_path, 
                                                     subset, '*mix.wav'))
            
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
            
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                mix_id = mix_id[:-4]
                
                speech_ref_file = os.path.join(os.path.dirname(input_file), 
                                               mix_id + '_speech.wav')
                assert os.path.isfile(speech_ref_file)
                
                output_file = os.path.join(reverberant_librichime5_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
            
                speech_est, sr = sf.read(output_file)
                speech_ref, sr = sf.read(speech_ref_file)
                
                si_sdr = sisdr_metric.compute_sisdr(speech_est, speech_ref)
                comp_sig,comp_bak,comp_ovr = composite(speech_ref,speech_est,16000) 
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), si_sdr,comp_sig,comp_bak,comp_ovr]
                
                reverberant_librichime5_df.loc[len(reverberant_librichime5_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    
                    si_sdr = sisdr_metric.compute_sisdr(mix, speech_ref)
                    comp_sig,comp_bak,comp_ovr = composite(speech_ref,speech_est,16000)
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), si_sdr,comp_sig,comp_bak,comp_ovr]
                    
                    unprocessed_reverberant_librichime5_df.loc[len(unprocessed_reverberant_librichime5_df)] = row
        
        csv_dir = os.path.join(output_path, 'csv', 'reverberant-LibriCHiME-5')
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        csv_file = os.path.join(csv_dir, 'results.csv')
        
        reverberant_librichime5_df.to_csv(csv_file)
    
        if compute_input_scores:     
            csv_file = os.path.join(output_path, 'csv', 'reverberant-LibriCHiME-5', 
                                    'results_unprocessed.csv')
            unprocessed_reverberant_librichime5_df.to_csv(csv_file)
        
    # LibriMix
    
    if librimix_input_path is not None and librimix_subsets is not None:
    
        print('Compute results on LibriMix')
        
        librimix_output_path = os.path.join(output_path, 'audio', 'LibriMix')
        
        librimix_df = pd.DataFrame(columns=['subset', 
                                            'input_file_name', 
                                            'output_file_name', 
                                            'SI-SDR'])
        
        if compute_input_scores:
            unprocessed_librimix_df = pd.DataFrame(columns=['subset', 
                                                            'input_file_name', 
                                                            'output_file_name', 
                                                            'SI-SDR'])
        
        for subset in librimix_subsets:
            
            print(subset)
            
            input_file_list = glob.glob(os.path.join(librimix_input_path, 
                                                     subset, '*.wav'))
            
            for ind, input_file in tqdm(enumerate(input_file_list), 
                                        total=len(input_file_list)):
            
                mix_id = os.path.basename(input_file)
                mix_id = os.path.splitext(mix_id)[0]
                
                # read speech estimate
                output_file = os.path.join(librimix_output_path, subset, 
                                           mix_id + '_output' + '.wav')
                assert os.path.isfile(output_file)
            
                speech_est, sr = sf.read(output_file)
                
                # read speech reference
                if subset=='Libri2Mix/wav16k/max/test/mix_single':
                    sources = ['s1']
                elif subset=='Libri2Mix/wav16k/max/test/mix_both':
                    sources = ['s1', 's2']
                elif subset=='Libri3Mix/wav16k/max/test/mix_both':
                    sources = ['s1', 's2', 's3']
                
                speech_ref_list = []
                for source in sources:
                    
                    speech_ref_file = os.path.join(os.path.dirname(os.path.dirname(input_file)), 
                                                   source, mix_id + '.wav')
                    assert os.path.isfile(speech_ref_file)
                    
                    speech_ref_list.append(sf.read(speech_ref_file)[0])
                
                speech_ref = np.zeros_like(speech_ref_list[0])
                for s in speech_ref_list:
                    speech_ref += s    
                
                si_sdr = sisdr_metric.compute_sisdr(speech_est, speech_ref)
                
                row = [subset, os.path.basename(input_file), 
                       os.path.basename(output_file), si_sdr]
                
                librimix_df.loc[len(librimix_df)] = row
                
                if compute_input_scores:
                    
                    mix, sr = sf.read(input_file)
                    
                    si_sdr = sisdr_metric.compute_sisdr(mix, speech_ref)
                    
                    row = [subset, os.path.basename(input_file), 
                           os.path.basename(input_file), si_sdr]
                    
                    unprocessed_librimix_df.loc[len(unprocessed_librimix_df)] = row

        csv_dir = os.path.join(output_path, 'csv', 'LibriMix')
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
        csv_file = os.path.join(csv_dir, 'results.csv')
                            
        librimix_df.to_csv(csv_file)
    
        if compute_input_scores:        
            csv_file = os.path.join(output_path, 'csv', 'LibriMix', 'results_unprocessed.csv')
            unprocessed_librimix_df.to_csv(csv_file)

#%%

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-baseline", help="run the baseline",
                        action="store_true")
    
    parser.add_argument("--model", help="model to run/evaluate",
                        choices=['sudo-rm-rf', 'remixit', 'remixit-vad','cmgan'])
    
    parser.add_argument("--model_path", help="path to model baseline")
    parser.add_argument("--output_path", help="path to output directory")
    parser.add_argument("--eval-baseline", help="compute the baseline output scores",
                        action="store_true")
    
    parser.add_argument("--input-scores", help="additionally compute the performance metrics on the unprocessed noisy speech signals",
                        action="store_true")
    
    args = parser.parse_args()
    
    model = args.model
    output_path = args.output_path
    output_path = os.path.join(output_path, model)
    
    if model == 'sudo-rm-rf':
        checkpoint = 'pretrained_checkpoints/libri1to3mix_supervised_teacher_w_mixconsist.pt'
    elif model == 'remixit':
        checkpoint = 'pretrained_checkpoints/remixit_chime_adapted_student.pt'
    elif model == 'remixit-vad':
        checkpoint = 'pretrained_checkpoints/remixit_chime_adapted_student_using_vad.pt'
    elif model == 'cmgan':
        #checkpoint = '/mnt/parscratch/users/acp20glc/sb_results/CHIME2023/CMGAN/dnsmos_sig-NOISY-w1.0-chime-7234-backup/save/CKPT+2023-06-09+01-23-32+00/model.ckpt'
        #checkpoint = '/mnt/parscratch/users/acp20glc/sb_results/CHIME2023/CMGAN/dnsmos_sig-NOISY-w1.0-chime-7234/save/CKPT+2023-06-12+12-59-47+00/model.ckpt'
        #checkpoint = '/home/acp20glc/projects/CHiME2023/chime_results/no-discrim-chime-7234/save/CKPT+2023-06-13+06-29-57+00/model.ckpt'
        #checkpoint = '/home/acp20glc/projects/CHiME2023/chime_results/dnsmos_sig-NOISY-w1.0-chime-7234/save/CKPT+2023-06-12+12-59-47+00/model.ckpt'
        #checkpoint = '/home/acp20glc/projects/CHiME2023/chime_results/no-discrim-chime-7234/save/CKPT+2023-06-13+09-20-23+00/model.ckpt'
        #checkpoint = '/mnt/parscratch/users/acp20glc/sb_results/CHIME2023_new2/CMGAN/dnsmos_sig-NOISY-w1.0-chime-7234/save/CKPT+2023-08-03+09-03-10+00/model.ckpt'
        checkpoint = args.model_path
    else:
        raise ValueError('Unknown model')
    
    # run baseline
    if args.run_baseline:
        
        run_CMGAN(checkpoint=checkpoint, 
                     output_path=output_path,
                     datasets=[
                         'chime-5', 
                         'reverberant-librichime-5',
                          'librimix'
                         ],
                     chime5_input_path=chime5_input_path, 
                     chime5_subsets=chime5_subsets_run, 
                     reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                     reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                     librimix_input_path=librimix_input_path,
                     librimix_subsets=librimix_subsets)
    
    # compute scores
    if args.eval_baseline and args.input_scores:
        
        compute_metrics(output_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets_eval, 
                        reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                        reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                        librimix_input_path=librimix_input_path,
                        librimix_subsets=librimix_subsets, 
                        compute_input_scores=True)
        
    elif args.eval_baseline:
        
        compute_metrics(output_path=output_path,
                        chime5_input_path=chime5_input_path, 
                        chime5_subsets=chime5_subsets_eval, 
                        reverberant_librichime5_input_path=reverberant_librichime5_input_path, 
                        reverberant_librichime5_subsets=reverberant_librichime5_subsets,
                        librimix_input_path=librimix_input_path,
                        librimix_subsets=librimix_subsets, 
                        compute_input_scores=False)
        
        
    