"""!
@brief Final evaluation of a pre-trained checkpoint model on the CHiME-5 and LibriCHiME-5 data.
@author Efthymios Tzinis {etzinis2@illinois.edu}
@author Mostafa Sadeghi {mostafa.sadeghi@inria.fr}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse
import torch
import numpy as np
import pyloudnorm as pyln

from tqdm import tqdm
from pprint import pprint
import pickle
import os

import baseline.utils.mixture_consistency as mixture_consistency
import baseline.models.improved_sudormrf as improved_sudormrf
import baseline.metrics.dnnmos_metric as dnnmos_metric
import baseline.metrics.sisdr_metric as sisdr_metric
import baseline.dataset_loaders.chime as chime
import baseline.dataset_loaders.libri1to3chime as libri1to3chime
from models.CMGAN.generator import TSCNet
def get_args():
    """! Command line parser"""
    parser = argparse.ArgumentParser(description="Final evaluation Argument Parser")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="""The absolute path of a pre-trained separation model
            that will be used for warm start for the teacher network.""",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="""The target dataset for evaluation.""",
        default='chime',
    )
    parser.add_argument(
        "--save_results_dir",
        type=str,
        help="""The absolute path for saving the full eval results file.""",
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="""The experiment's name.""",
        default='',
    )
    parser.add_argument(
        "--evaluate_only_input_mixture",
        action="store_true",
        help="""Whether to evaluate on the input mixture only.""",
        default=False,
    )
    parser.add_argument(
        "--si_sdr_librichime",
        action="store_true",
        help="""Compute SI-SDR metric on the LibriCHiME dev set.""",
        default=False,
    )
    return parser.parse_args()

# For DNS-MOS computations, fixed_n_sources = 1, i.e., only 1-speaker files
def get_librichime_generator():
    data_loader = libri1to3chime.Dataset(
        sample_rate=16000, fixed_n_sources=1,
        timelength=-1., augment=False,
        zero_pad=False, split='dev',
        normalize_audio=False, n_samples=-1)

    return data_loader.get_generator(batch_size=1, num_workers=1)

def get_chime_generator():
    data_loader = chime.Dataset(
        sample_rate=16000, fixed_n_sources=1,
        timelength=-1., augment=False, use_vad=False,
        zero_pad=False, split='dev', get_only_active_speakers=False,
        normalize_audio=False, n_samples=-1)
    return data_loader.get_generator(batch_size=1, num_workers=1)


def load_sudo_rm_rf_model(path):
    model = improved_sudormrf.SuDORMRF(
        out_channels=256,
        in_channels=512,
        num_blocks=8,
        upsampling_depth=7,
        enc_kernel_size=81,
        enc_num_basis=512,
        num_sources=2,
    )
    # You can load the state_dict as here:
    model.load_state_dict(torch.load(path))
    print(f"Fetched model from: {path}")
    return model

def load_cmgan_model(path):
    model = TSCNet()
    model.load_state_dict(torch.load(path))
    print(f"Fetched model from: {path}")
    return model



if __name__ == "__main__":
    fs = 16000
    meter = pyln.Meter(fs)
    args = get_args()
    hparams = vars(args)
    if hparams['dataset'] == 'chime':
        test_generator = get_chime_generator()
        flag = False
    elif hparams['dataset'] == 'librichime':
        test_generator = get_librichime_generator()
        flag = True
    else:
        ValueError('Unknown dataset. It should be either chime or librichime.')
        
    if not hparams['evaluate_only_input_mixture']:
        model = load_sudo_rm_rf_model(hparams['model_checkpoint'])
        model = model.cuda()
        model.eval()
    else:
        model = None
    test_tqdm_gen = tqdm(enumerate(test_generator), desc='Eval on 16kHz ' + hparams['dataset'] + ' 1 speaker')
    res_dic = {
            "sig_mos": [],
            "bak_mos": [],
            "ovr_mos": [],
            "si_sdr":  [],
            "si_sdri": [],
    }
    gen_len = len(test_generator)
    with torch.no_grad():
        for j, input_argument in test_tqdm_gen:
            if flag:
                speakers, noise = input_argument
                gt_speaker_mix = speakers.sum(1, keepdims=True) 
                mixture = noise + gt_speaker_mix
            else:
                mixture = input_argument
                    
            if hparams['evaluate_only_input_mixture']:
                s_est_speech = mixture[0].cpu().numpy().squeeze()
            else:
                file_length = mixture.shape[-1]
                min_k = int(np.ceil(np.log2(file_length/16000)))
                padded_length = 2**max(min_k, 1) * 16000

                input_mix = torch.zeros((1, padded_length), dtype=mixture.dtype)
                input_mix[..., :file_length] = mixture

                input_mix = input_mix.unsqueeze(1).cuda()
                input_mix_std = input_mix.std(-1, keepdim=True)
                input_mix_mean = input_mix.mean(-1, keepdim=True)
                input_mix = (input_mix - input_mix_mean) / (input_mix_std + 1e-9)

                student_estimates = model(input_mix)
                student_estimates = mixture_consistency.apply(student_estimates, input_mix)

                s_est_speech = student_estimates[0, 0, :file_length].detach().cpu().numpy().squeeze()
                
            # SI-SDR on dev set of LibriCHiME
            if hparams['si_sdr_librichime'] and flag:
                gt_speaker_mix = gt_speaker_mix.cpu().numpy().squeeze()
                si_sdr = sisdr_metric.compute_sisdr(s_est_speech, gt_speaker_mix)

                si_sdri = si_sdr - sisdr_metric.compute_sisdr(
                    input_mix[0, 0, :file_length].cpu().numpy().squeeze(), gt_speaker_mix)

                res_dic["si_sdr"].append(si_sdr)
                res_dic["si_sdri"].append(si_sdri)
            else:    
                # Peak normalization to 0.7
                s_est_speech -= s_est_speech.mean()
                s_est_speech /= np.abs(s_est_speech).max() + 1e-9
                s_est_speech *= 0.7

                # Loudness normalization to LUFS = -30:
                loudness = meter.integrated_loudness(s_est_speech)
                s_est_speech = pyln.normalize.loudness(s_est_speech, loudness, -30.0)


                # DNS-MOS
                dnsmos_res_dic = dnnmos_metric.compute_dnsmos(s_est_speech, fs=16000)
                for k, v in dnsmos_res_dic.items():
                    res_dic[k].append(v)
                ovrl_mos_avg = round(np.mean(res_dic["ovr_mos"]), 2)
                bak_mos_avg = round(np.mean(res_dic["bak_mos"]), 2)
                sig_mos_avg = round(np.mean(res_dic["sig_mos"]), 2)
                test_tqdm_gen.set_description(
                    f"Avg OVRL MOS: {ovrl_mos_avg}, BAK: {bak_mos_avg}, SIG: {sig_mos_avg} {j}/{gen_len}")

    aggregate_results = {}
    
    for k, values in res_dic.items():
        mean_metric = np.mean(values)
        median_metric = np.median(values)
        std_metric = np.std(values)
        aggregate_results[k] = {'mean': mean_metric, 'median': median_metric, 'std': std_metric}
            
    pprint(aggregate_results)
    if hparams["evaluate_only_input_mixture"]:
        model_name = 'unprocessed'
    else:
        model_name = os.path.basename(hparams['model_checkpoint'])

    saved_name = model_name + '_full_eval_' + hparams['dataset'] + '_results_ '  + hparams['experiment_name'] + '.pkl'
    if hparams['save_results_dir'] is None:
        save_path = os.path.join('/tmp', saved_name)
    else:
        save_path = os.path.join(hparams['save_results_dir'], saved_name)

    with open(save_path, 'wb') as handle:
        pickle.dump(aggregate_results, handle, protocol=pickle.HIGHEST_PROTOCOL)