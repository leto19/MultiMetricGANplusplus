#!/usr/bin/env/python3
"""
Recipe for training a speech enhancement system with the Voicebank dataset.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Szu-Wei Fu 2020
 * Peter Plantinga 2021
MultiMetricConformerGAN+/+ additions by George Close 2023
"""
import os
import sys
import shutil
import torch
import torchaudio
import speechbrain as sb
from pesq import pesq
from enum import Enum, auto
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.nnet.loss.si_snr_loss import si_snr_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler
import pickle
from metrics.dnnmos_metric import compute_dnsmos
import torchinfo
import numpy as np
import pyloudnorm as pyln
from utils.dataset_setup import supervised_setup,unsupervised_setup

meter = pyln.Meter(16000)

def pesq_eval(pred_wav, target_wav):
    """Normalized PESQ (to 0-1)"""
    if torch.equal(pred_wav,target_wav):# avoid a reduntant cal to the pesq function since we know it will be 1 in this case
        return 1.0 
    try: 
        p = (pesq(fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb") + 0.5) /5
    except:
        p = 0.3 # handle the case that PESQ fails 

    return p


def ddnsmos_eval_sig(pred_wav,target_wav):
    #print("before ln",np.isnan(pred_wav).any())

    pred_wav = apply_loudness_normalization(pred_wav.numpy())
    pred_wav = np.nan_to_num(pred_wav)
    #print("after ln",np.isnan(pred_wav).any())
    p = compute_dnsmos(pred_wav)["sig_mos"] /5
    return float(p)

def ddnsmos_eval_bak(pred_wav,target_wav):
    pred_wav = apply_loudness_normalization(pred_wav.numpy())
    pred_wav = np.nan_to_num(pred_wav)

    p = compute_dnsmos(pred_wav)["bak_mos"] /5
    return float(p)

def ddnsmos_eval_ovr(pred_wav,target_wav):
    pred_wav = apply_loudness_normalization(pred_wav.numpy())
    pred_wav = np.nan_to_num(pred_wav)

    p = compute_dnsmos(pred_wav)["ovr_mos"] /5
    return float(p)

def ddnsmos_eval_all(pred_wav,target_wav):
    pred_wav = apply_loudness_normalization(pred_wav)

    p = compute_dnsmos(pred_wav.numpy())
    p_sig = float(p["sig_mos"] /5)
    p_bak = float(p["bak_mos"] /5)
    p_ovr = float(p["ovr_mos"] /5)

    return p_sig,p_bak,p_ovr


def apply_loudness_normalization(s_est_speech, target_lufs = -30.0):
        s_est_speech = s_est_speech.squeeze()
        # Peak normalization to 0.7, just to make sure we don't get loudness=-inf
        s_est_speech -= s_est_speech.mean()
        s_est_speech /= np.abs(s_est_speech).max() + 1e-9
        s_est_speech *= 0.7
        # Loudness normalization
        loudness = meter.integrated_loudness(s_est_speech)
        s_est_speech = pyln.normalize.loudness(s_est_speech, loudness, target_lufs)
        return s_est_speech

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

class SubStage(Enum):
    """For keeping track of training stage progress"""

    GENERATOR = auto()
    DEGENERATOR =auto()
    CURRENT = auto()
    HISTORICAL = auto()


class MetricGanBrain(sb.Brain):
    def load_history(self):
        print("loading historical set!")

        if os.path.isfile(self.hparams.historical_file):
            with open(self.hparams.historical_file, "rb") as fp:  # Unpickling
                self.historical_set = pickle.load(fp)


    def compute_feats_mag(self, wavs):
        """Feature computation pipeline"""
        #print("WAVS",wavs.shape)
        if wavs.shape[0] != self.hparams.train_N_batch and wavs.shape[0] != self.hparams.valid_N_batch:
            wavs = wavs.unsqueeze(0)
        feats = self.hparams.compute_STFT(wavs)

        #print("FEATS",feats.shape) # B * T * F * 2
        #print(feats.shape)
        feats_real = feats[:,:,:,0].unsqueeze(1)
        feats_imag = feats[:,:,:,1].unsqueeze(1)
        #print("feats_real",feats_real.shape)
        #print("feats_imag",feats_imag.shape)
        #feats_real = torch.add(feats_real,1e-10)
        #feats_imag = torch.add(feats_imag,1e-10)
        #print("feats_real",feats_real.shape)
        #print("feats_imag",feats_imag.shape)
        feats_mag = torch.sqrt(feats_real**2 + feats_imag**2)
        
        #print("feats_mag",feats_mag.shape)
        return feats_mag.squeeze(1)

    def compute_feats_mag_no_power(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs) # B * T * F * 2 
        feats = spectral_magnitude(feats, power=0.5) # B * T * F
        feats = torch.log1p(feats)

        return feats

    

    def compute_feats_real_imag(self, wavs):
        """Feature computation pipeline"""
        #print("WAVS",wavs.shape)
        feats = self.hparams.compute_STFT(wavs)
        #print("FEATS",feats.shape) # B * T * F * 2 
        feats_spec = power_compress(feats) # B * 2 * T * F
        #print("FEATS_SPEC",feats_spec.shape)
        feats_real = feats_spec[:,0,:,:]
        feats_imag = feats_spec[:,1,:,:]

        return feats_spec,feats_real,feats_imag

    def format_batch(self,batch):
        if len(batch) == 3:
            wav_id,clean_wavs,noise = batch
            #print(wav_id)
            #sum the clean speech channels
           # print(clean_wavs.shape,noise.shape)
            clean_wavs = torch.sum(clean_wavs,1,keepdim=True).to(self.device)
            # add the noise
            noisy_wavs = clean_wavs + noise.to(self.device)
            clean_wavs = clean_wavs.squeeze(1)
            noisy_wavs = noisy_wavs.squeeze(1)
            #print(clean_wavs.shape,noisy_wavs.shape)
            # apply mean and variance normalisation 
            #noisy_wavs_std = noisy_wavs.std(-1, keepdim=True)
            #noisy_wavs_mean = noisy_wavs.mean(-1, keepdim=True)
            #noisy_wavs = (noisy_wavs - noisy_wavs_mean) / (noisy_wavs_std+ 1e-9)

            
            return wav_id,noisy_wavs,clean_wavs,torch.ones(clean_wavs.shape[0]).to(self.device)
        
        else: # we're dealing with real CHiME data with no reference:
            
            wav_id,noisy_wavs = batch
            print(wav_id)
            noisy_wavs = noisy_wavs.to(self.device)

            #noisy_wavs_std = noisy_wavs.std(-1, keepdim=True)
            #noisy_wavs_mean = noisy_wavs.mean(-1, keepdim=True)
            #noisy_wavs = (noisy_wavs - noisy_wavs_mean) / (noisy_wavs_std+ 1e-9)
            #create a dummy tensor for the clean wavs (for consistancy)
            clean_wavs = noisy_wavs

            return wav_id,noisy_wavs,clean_wavs,torch.ones(clean_wavs.shape[0]).to(self.device)

    def compute_forward(self, batch, stage):
        "Given an input batch computes the enhanced signal"
        if self.sub_stage != SubStage.HISTORICAL:
            wav_id,noisy_wavs,clean_wavs,lens = self.format_batch(batch)

        if self.sub_stage == SubStage.HISTORICAL:
            predict_wav, lens = batch.enh_sig
            return predict_wav
        elif stage == SubStage.DEGENERATOR:
            if self.hparams.degen_mode =="NOISY":
                #noisy_wav, lens = batch.noisy_sig
                noisy_wavs = noisy_wavs
            if self.hparams.degen_mode  == "CLEAN":
                noisy_wavs = clean_wavs
            noisy_spec = self.compute_feats_mag_no_power(noisy_wavs)
            #print("DEGEN NOISY SPEC:",noisy_spec.shape)
            mask = self.modules.degenerator(noisy_spec, lengths=lens)
            #print(mask.shape)
            mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
            predict_spec = torch.mul(mask,noisy_spec )
            #predict_spec = power_uncompress_2(predict_spec)
           
            predict_wav = self.hparams.resynth(
                torch.expm1(predict_spec), noisy_wavs
            )
            #print(torch.isnan(predict_wav).any())
            # check for nan values in the predicted wavs:
            
            return predict_wav

        else:
            # apply mean and variance normalisation 
            noisy_wavs_std = noisy_wavs.std(-1, keepdim=True)
            noisy_wavs_mean = noisy_wavs.mean(-1, keepdim=True)
            noisy_wavs = (noisy_wavs - noisy_wavs_mean) / (noisy_wavs_std+ 1e-9)
            
            noisy_spec,noisy_real,noisy_imag = self.compute_feats_real_imag(noisy_wavs)
            #print(noisy_spec.shape)
            predict_real,predict_imag = self.modules.generator(noisy_spec)
            #print(predict_real.shape,predict_imag.shape)

            predict_spec_uncompress = power_uncompress(predict_real, predict_imag).squeeze(1)
            #print(predict_spec_uncompress.shape)

            predict_wav = self.hparams.compute_ISTFT(predict_spec_uncompress)
            #print(predict_wav.shape)
            return predict_wav,predict_real,predict_imag
            
    def compute_objectives(self, predictions, batch, stage, optim_name=""):
        #print(optim_name,self.sub_stage)
        "Given the network predictions and targets compute the total loss"
        if predictions is not None and len(predictions) == 3:
            predict_wav,predict_real,predict_imag = predictions
            #print(predict_wav.shape,predict_real.shape,predict_imag.shape)
        else:
            predict_wav = predictions
        predict_spec = self.compute_feats_mag(predict_wav)
        #print(self.sub_stage)

        if self.sub_stage != SubStage.HISTORICAL:
            #print(len(batch))
            
            wav_id,noisy_wavs,clean_wavs,lens = self.format_batch(batch)
            
            if self.supervised == False and torch.equal(noisy_wavs,clean_wavs):
                #print("using teacher!")
                # in unsupervised training, use the teacher to get
                # the 'clean' estimate
                #print(clean_wavs.shape)
                noisy_spec,noisy_real,noisy_imag = self.compute_feats_real_imag(noisy_wavs)
                clean_real,clean_imag = self.modules.teacher(noisy_spec)
                clean_spec_uncompress = power_uncompress(clean_real, clean_imag).squeeze(1)
                clean_wavs = self.hparams.compute_ISTFT(clean_spec_uncompress)
                #print(clean_wavs.shape)
            clean_spec = self.compute_feats_mag(clean_wavs)
            ids = self.compute_ids(wav_id, optim_name)
            #print(ids)
            mse_cost = self.hparams.compute_cost(predict_spec, clean_spec, lens)
            # One is real, zero is fake
        else:
            predict_wav = predict_wav.to(self.device)
            batch = batch.to(self.device)

        if optim_name == "generator":
            
            if self.hparams.target_metric == "SISDR_combo":
                # if we're predicting SI-SDR, we don't need a target score for it
                target_score = torch.ones(self.batch_size, 2, device=self.device)
                est_score = self.est_score(predict_wav)
                est_si_sdr_score = est_score[:,2]
                est_score = est_score[:,:2]
                #print(est_si_sdr_score)
                #print(est_score)
            else:                

                target_score = torch.ones(self.batch_size, 3, device=self.device)
                est_score = self.est_score(predict_wav)
          

            if self.supervised:
                self.mse_metric.append(
                    ids, predict_spec, clean_spec, lens, reduction="batch"
                )
                time_score = torch.mean(torch.abs(predict_wav - clean_wavs))
                sisdr_score = si_snr_loss(predict_wav,clean_wavs,lens)
                #print("GAN score: %s"%est_score)
            else:
                self.mse_metric.append(
                    ids, predict_spec, clean_spec, lens, reduction="batch"
                )
                time_score = torch.mean(torch.abs(predict_wav - clean_wavs))
                sisdr_score = si_snr_loss(predict_wav,clean_wavs,lens)
                #print("unsupervised GAN score: %s"%est_score)



            
        elif optim_name == "degenerator":
            if self.hparams.target_metric == "SISDR_combo":
                # if we're predicting SI-SDR, we don't need a target score for it
                target_score = torch.ones(self.batch_size, 2, device=self.device)
                target_score = target_score * self.hparams.degraded_target
                est_score = self.est_score(predict_wav)
                est_si_sdr_score_degen = est_score[:,2]
                est_score = est_score[:,:2]
                print(est_si_sdr_score_degen)
                print(est_score)
            else:
                target_score = torch.ones(self.batch_size, 3, device=self.device)
                #we set the target of the degenerator to be less than 1 
                target_score = target_score * self.hparams.degraded_target
                #print(target_score)
                est_score = self.est_score(predict_wav)
            #print(est_score)
            #self.mse_metric.append(
            #    ids, predict_spec, clean_spec, lens, reduction="batch"
            #)

        # D Learns to estimate the scores of clean speech
        elif optim_name == "D_clean":

            if "dns" in self.hparams.target_metric or "_combo" in self.hparams.target_metric:
                target_score = self.score_multi(ids, clean_wavs, clean_wavs, lens)
            else:
                target_score = torch.ones(self.batch_size, 1, device=self.device)
            
            est_score = self.est_score(clean_wavs)

        # D Learns to estimate the scores of enhanced speech
        elif optim_name == "D_enh" and self.sub_stage == SubStage.CURRENT:
            #print("D Learns to estimate the scores of enhanced speech")
            #input(">>>")
            target_score = self.score_multi(ids, predict_wav, clean_wavs, lens)
            est_score = self.est_score(predict_wav) 
      
            # Write enhanced wavs during discriminator training, because we
            # compute the actual score here and we can save it
            
            self.write_wavs(wav_id, ids, predict_wav, target_score, lens)

        elif optim_name == "D_denh" and self.sub_stage == SubStage.CURRENT:
            target_score = self.score_multi(ids, predict_wav, clean_wavs, lens)
            est_score = self.est_score(predict_wav)
           
            # Write enhanced wavs during discriminator training, because we
            # compute the actual score here and we can save it
            self.write_wavs(wav_id, ids, predict_wav, target_score, lens)



        # D Relearns to estimate the scores of previous epochs
        elif optim_name == "D_enh" and self.sub_stage == SubStage.HISTORICAL:
            target_score = torch.tensor(batch.score).float().to(self.device)
            est_score = self.est_score(predict_wav)
            
        elif optim_name == "D_denh" and self.sub_stage == SubStage.HISTORICAL:

            target_score = batch.score.unsqueeze(1).float()

            est_score = self.est_score(predict_wav)
        # D Learns to estimate the scores of noisy speech
        elif optim_name == "D_noisy":
            target_score = self.score_multi(ids, noisy_wavs, clean_wavs, lens)
            est_score = self.est_score(noisy_wavs)
     
            # Save scores of noisy wavs
            self.save_noisy_scores(ids, target_score.cpu().numpy())
        #print(stage)
        # if "D_" in optim_name:
        #     print(optim_name)
        #     print("target scores:",target_score)
        #     print("predicted_scores:",est_score)

        if stage == sb.Stage.TRAIN:
            # Compute the cost
            adv_cost = self.hparams.compute_cost(est_score, target_score)
            #print(optim_name,stage)
            if optim_name == "generator" or optim_name=="degenerator":
                if optim_name == "generator": # for the generator, use the other loss terms
                    #[0.1, 0.9, 0.2, 0.05]
                    #print("GAN loss: %s\nMAG loss: %s\nRI loss: %s\nTIME loss: %s"% (adv_cost,mag_score,ri_score,time_score))

                    
                    # adv_cost =  0.1 * ri_score + 0.9 * mag_score + 0.2 * time_score + 0.05 * adv_cost
                    if self.supervised:
                        adv_cost =  time_score  + sisdr_score + adv_cost
                    else:
                        
                        adv_cost =  time_score  + sisdr_score + adv_cost
                        #adv_cost =  0.1* sisdr_score + adv_cost

                    self.metrics["G"].append(adv_cost.detach())
                else:
                    self.metrics["DG"].append(adv_cost.detach())
            else:
                #print("updating D loss tracker!")
                #input(">>>")
                self.metrics["D"].append(adv_cost.detach())
        #print("METRICS AFTER")
        #print(self.metrics)

        # On validation data compute scores
        if stage != sb.Stage.TRAIN:
            adv_cost = mse_cost
            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                wav_id, predict_wav, clean_wavs, lens, reduction="batch"
            )
            self.pesq_metric.append(
                wav_id, predict=predict_wav, target=clean_wavs, lengths=lens
            )
            self.sisdr_metric.append(
                wav_id, predict_wav, clean_wavs, lens, reduction="batch"
            )
            self.dnsmos_sig_metric.append(
                wav_id,predict=predict_wav,target=clean_wavs, lengths=lens
            )
            self.dnsmos_bak_metric.append(
                wav_id,predict=predict_wav,target=clean_wavs, lengths=lens
            )
            self.dnsmos_ovr_metric.append(
                wav_id,predict=predict_wav,target=clean_wavs, lengths=lens
            )
            # Write wavs to file, for evaluation
            lens = lens * clean_wavs.shape[1]
            for name, pred_wav, length in zip(wav_id, predict_wav, lens):
                name += ".wav"
                enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                torchaudio.save(
                    enhance_path,
                    torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                    16000,
                )
            
            for name, clean_wav, length in zip(wav_id, clean_wavs, lens):
                    name += "_gt_mix.wav"
                    enhance_path = os.path.join(
                        self.hparams.enhanced_folder, name
                    )
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(clean_wav[: int(length)].cpu(), 0),
                        16000,
                    )
            for name, noisy_wav, length in zip(wav_id, noisy_wavs, lens):
                name += "_noisy_mix.wav"
                enhance_path = os.path.join(
                    self.hparams.enhanced_folder, name
                )
                torchaudio.save(
                    enhance_path,
                    torch.unsqueeze(noisy_wav[: int(length)].cpu(), 0),
                    16000,
                )

        # we do not use mse_cost to update model
        return adv_cost

    def compute_ids(self, batch_id, optim_name):
        """Returns the list of ids, edited via optimizer name."""
        if optim_name == "D_enh":
            return [f"{uid}@{self.epoch}" for uid in batch_id]
        elif optim_name == "D_denh":
            return [f"{uid}@{self.epoch}_d" for uid in batch_id]
        return batch_id

    def save_noisy_scores(self, batch_id, scores):
        #print("                         SAVE NOISY SCORES!!")
        for i, score in zip(batch_id, scores):
            #print(i,score)
            self.noisy_scores[i] = score

    def score(self, batch_id, deg_wav, ref_wav, lens):
        """Returns actual metric score, either pesq or stoi

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        deg_wav : torch.Tensor
            The degraded waveform to score
        ref_wav : torch.Tensor
            The reference waveform to use for scoring
        length : torch.Tensor
            The relative lengths of the utterances
        """
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]

        if len(new_ids) == 0:
            pass
        elif self.hparams.target_metric == "pesq":
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[new_ids].detach(),
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device,
            )
        elif self.hparams.target_metric == "stoi":
            self.target_metric.append(
                [batch_id[i] for i in new_ids],
                deg_wav[new_ids],
                ref_wav[new_ids],
                lens[new_ids],
                reduction="batch",
            )
            score = torch.tensor(
                [[-s] for s in self.target_metric.scores], device=self.device,
            )
        elif self.hparams.target_metric == "dnsmos_sig" or self.hparams.target_metric == "dnsmos_bak" or self.hparams.target_metric == "dnsmos_ovr":
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[new_ids].detach(),
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device,
            )
        elif self.hparams.target_metric == "dnsmos_all" or "pesq_combo" in self.hparams.target_metric:
            return self.score_multi(batch_id,deg_wav,ref_wav,lens)
        else:
            raise ValueError("Expected 'pesq' or 'stoi' for target_metric")

        # Clear metric scores to prepare for next batch
        self.target_metric.clear()

        # Combine old scores and new
        final_score = []
        for i, d in enumerate(batch_id):
            if d in self.historical_set:
                final_score.append([self.historical_set[d]["score"]])
            elif d in self.noisy_scores:
                final_score.append([self.noisy_scores[d]])
            else:
                final_score.append([score[new_ids.index(i)]])

        return torch.tensor(final_score, device=self.device)



    def score_multi(self, batch_id, deg_wav, ref_wav, lens):
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]
        if len(new_ids) == 0:
            # Combine old scores and new
            pass
        else:
            metric_out = []
            for metric in self.target_metric:
                #print(metric)
                if metric.metric == si_snr_loss:
                    metric.append(
                    [batch_id[i] for i in new_ids],
                    deg_wav[new_ids],
                    ref_wav[new_ids],
                    lens[new_ids],
                    reduction="batch",
                    )
                    score = [-s for s in metric.scores]
                else:
                    metric.append(
                    ids=[batch_id[i] for i in new_ids],
                    predict=deg_wav[new_ids].detach(),
                    target=ref_wav[new_ids].detach(),
                    lengths=lens[new_ids])
                    score = [s for s in metric.scores]
                
                #print(score)
                metric_out.append(score)
                #print(metric_out)
                #input(">>>")
                # Clear metric scores to prepare for next batch
                metric.clear()
            metric_out = list(zip(*metric_out))
        #print(">>>")
        # Combine old scores and new
        final_score = []
        for i, d in enumerate(batch_id):
            #print(i,d)
            if d in self.historical_set:
                final_score.append([self.historical_set[d]["score"]])
            elif d in self.noisy_scores:
                final_score.append([self.noisy_scores[d]])
            else:
                final_score.append([metric_out[new_ids.index(i)]])
        

        out_tensor = torch.tensor(final_score, device=self.device).squeeze()

    
        return out_tensor

    def est_score(self, deg_wav):
        """Returns score as estimated by discriminator

        Arguments
        ---------
        deg_spec : torch.Tensor
            The spectral features of the degraded utterance
        ref_spec : torch.Tensor
            The spectral features of the reference utterance
        """
        #combined_spec = torch.cat(
        #    [deg_spec.unsqueeze(1), ref_spec.unsqueeze(1)], 1
        #)
        
        return self.modules.discriminator(deg_wav)

    def write_wavs(self, clean_id, batch_id, wavs, score, lens):
        """Write wavs to files, for historical discriminator training

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        wavs : torch.Tensor
            The wavs to write to files
        score : torch.Tensor
            The actual scores for the corresponding utterances
        lens : torch.Tensor
            The relative lengths of each utterance
        """
        #print("score in write_wavs:",score)
        if self.hparams.history_portion == 0:
            return
        #print(clean_id,batch_id,wavs.shape,score,lens)
        lens = lens * wavs.shape[1]
        record = {}
        for i, (cleanid, name, pred_wav, length) in enumerate(
            zip(clean_id, batch_id, wavs, lens)
        ):
            path = os.path.join(self.hparams.MetricGAN_folder, name + ".wav")
            data = torch.unsqueeze(pred_wav[: int(length)].cpu(), 0)
            #print(path,data.shape,score)
            torchaudio.save(path, data, self.hparams.Sample_rate)

            # Make record of path and score for historical training

            score_l1 = float(score[i][0])
            score_l2 = float(score[i][1])
            score_l3 = float(score[i][2])
            score_l = [score_l1,score_l2,score_l3]
            #clean_path = os.path.join(
            #    self.hparams.train_clean_folder, cleanid + ".wav"
            #)
            record[name] = {
                "enh_wav": path,
                "score": score_l,
                #"clean_wav": clean_path
            }
            
            #print("record saved!\n",record[name])
        # Update records for historical training
        self.historical_set.update(record)
        with open(self.hparams.historical_file, "wb") as fp:  # Pickling
            pickle.dump(self.historical_set, fp)


    

    def fit_batch(self, batch):
        "Compute gradients and update either D or G based on sub-stage."
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        de_predictions = self.compute_forward(batch,SubStage.DEGENERATOR)
        loss_tracker = 0

        if self.supervised:
            mode_list = ["clean", "enh", "noisy","denh"]
        else:
            mode_list = ["enh", "noisy","denh"]
        if self.sub_stage == SubStage.CURRENT:

            for mode in mode_list:
                
                if mode == "denh":
                    loss = self.compute_objectives(
                                            de_predictions, batch, sb.Stage.TRAIN, f"D_{mode}"
                                        )
                else:
                    loss = self.compute_objectives(
                        predictions, batch, sb.Stage.TRAIN, f"D_{mode}"
                    )
                self.d_optimizer.zero_grad()
                loss.backward()
                if self.check_gradients(loss):
                    self.d_optimizer.step()
                #loss_tracker += loss.detach() / 3
                
                loss_tracker += loss.detach() / len(mode_list)

        elif self.sub_stage == SubStage.HISTORICAL:
            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "D_enh"
            )
            self.d_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.d_optimizer.step()
            loss_tracker += loss.detach()
        elif self.sub_stage == SubStage.GENERATOR:
            for name, param in self.modules.generator.named_parameters():
                if "Learnable_sigmoid" in name:
                    param.data = torch.clamp(
                        param, max=3.5
                    )  # to prevent gradient goes to infinity

            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "generator"
            )
            self.g_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.g_optimizer.step()
            loss_tracker += loss.detach()
        elif  self.sub_stage == SubStage.DEGENERATOR:
            #input("started degenerator training!")
            for name, param in self.modules.degenerator.named_parameters():
                if "Learnable_sigmoid" in name:
                    param.data = torch.clamp(
                        param, max=3.5
                    )  # to prevent gradient goes to infinity

            loss = self.compute_objectives(
                de_predictions, batch, sb.Stage.TRAIN, "degenerator"
            )
            self.gd_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.gd_optimizer.step()
            loss_tracker += loss.detach()


            
        return loss_tracker

    def train_degenerator(self):

        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch

        This method calls ``fit()`` again to train the discriminator
        before proceeding with generator training.
        """

        self.mse_metric = MetricStats(metric=self.hparams.compute_cost)
        
        if hasattr(self, 'metrics') and self.metrics["D"] is not []:
            self.metrics = {"G": [], "D": self.metrics["D"],"DG": []}
        else:
            self.metrics = {"G": [], "D": [],"DG": []}

        if stage == sb.Stage.TRAIN:
            if self.hparams.target_metric == "pesq":
                self.target_metric = MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False)
            elif self.hparams.target_metric == "stoi":
                self.target_metric = MetricStats(metric=stoi_loss)
            elif self.hparams.target_metric == "dnsmos_sig":
                self.target_metric = MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False)
            elif self.hparams.target_metric == "dnsmos_bak":
                self.target_metric = MetricStats(metric=ddnsmos_eval_bak, n_jobs=1,batch_eval=False)
            elif self.hparams.target_metric == "dnsmos_ovr":
                self.target_metric = MetricStats(metric=ddnsmos_eval_ovr, n_jobs=1,batch_eval=False)
            elif self.hparams.target_metric == "dnsmos_all": #SIG,BAK,OVRL
                self.target_metric = [
                       MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False),
                       MetricStats(metric=ddnsmos_eval_bak, n_jobs=1,batch_eval=False),
                       MetricStats(metric=ddnsmos_eval_ovr, n_jobs=1,batch_eval=False)
                ]
            elif self.hparams.target_metric == "pesq_combo": #SIG,BAK,PESQ
                self.target_metric = [
                       MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False),
                       MetricStats(metric=ddnsmos_eval_bak, n_jobs=1,batch_eval=False),
                       MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False)
                ]
            elif self.hparams.target_metric == "pesq_combo2": #SIG,OVRL,PESQ
                self.target_metric = [
                       MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False),
                       MetricStats(metric=ddnsmos_eval_ovr, n_jobs=1,batch_eval=False),
                       MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False)
                ]
            elif self.hparams.target_metric == "pesq_combo3": #BAK,OVRL,PESQ
                self.target_metric = [
                       MetricStats(metric=ddnsmos_eval_bak, n_jobs=1,batch_eval=False),
                       MetricStats(metric=ddnsmos_eval_ovr, n_jobs=1,batch_eval=False),
                       MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False)
                ]
            elif self.hparams.target_metric == "SISDR_combo":
                self.target_metric = [
                    MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False),
                    MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False),
                    MetricStats(metric=si_snr_loss)
                ]

            else:
                raise NotImplementedError(
                    "Right now we only support 'pesq' and 'stoi' and 'dnsmos_X and X_combo'"
                )

            # Train discriminator before we start generator training
            if self.sub_stage == SubStage.GENERATOR:
                self.epoch = epoch
                
                self.train_discriminator()
                self.sub_stage = SubStage.DEGENERATOR
                print("Degenerator training by current data...")
                self.train_degenerator()
                self.sub_stage = SubStage.GENERATOR
                print("Generator training by current data...")
                
        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(metric=pesq_eval, n_jobs=1,batch_eval=False)
            self.stoi_metric = MetricStats(metric=stoi_loss)
            self.sisdr_metric = MetricStats(metric=si_snr_loss)
            self.dnsmos_sig_metric = MetricStats(metric=ddnsmos_eval_sig, n_jobs=1,batch_eval=False)
            self.dnsmos_bak_metric = MetricStats(metric=ddnsmos_eval_bak, n_jobs=1,batch_eval=False)
            self.dnsmos_ovr_metric = MetricStats(metric=ddnsmos_eval_ovr, n_jobs=1,batch_eval=False)



    def train_discriminator(self):
        """A total of 3 data passes to update discriminator."""
        # First, iterate train subset w/ updates for clean, enh, noisy
        print("Discriminator training by current data...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

        # Next, iterate historical subset w/ updates for enh
        if self.historical_set:
            print("Discriminator training by historical data...")
            #print(len(self.historical_set))
            #input(">>>>")
            self.sub_stage = SubStage.HISTORICAL
            self.fit(
                range(1),
                self.historical_set,
                train_loader_kwargs={"batch_size": 20}
            )

            
    def on_stage_end(self, stage, stage_loss, epoch=None):
        
        "Called at the end of each stage to summarize progress"
        if self.sub_stage != SubStage.GENERATOR:
            return

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            g_loss = torch.tensor(self.metrics["G"])  # batch_size
            gd_loss = torch.tensor(self.metrics["DG"])
            d_loss = torch.tensor(self.metrics["D"])  # batch_size
            self.metrics = {"G": [], "D": [],"DG": []}
            print("Avg G loss: %.3f" % torch.mean(g_loss))
            print("Avg DG loss: %.3f" % torch.mean(gd_loss))

            print("Avg D loss: %.3f" % torch.mean(d_loss))
            print("MSE distance: %.3f" % self.mse_metric.summarize("average"))

        else:
            stats = {
                "MSE distance": stage_loss,
                "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                "stoi": -self.stoi_metric.summarize("average"),
                "SI-SDR": -self.sisdr_metric.summarize("average"),
                "dnsmos_sig": 5* self.dnsmos_sig_metric.summarize("average"),
                "dnsmos_bak": 5* self.dnsmos_bak_metric.summarize("average"),
                "dnsmos_ovr": 5* self.dnsmos_ovr_metric.summarize("average")

            }

        if stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "mse": stage_loss,
                    "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                    "stoi": -self.stoi_metric.summarize("average"),
                    "SI-SDR": -self.sisdr_metric.summarize("average"),
                    "dnsmos_sig": 5* self.dnsmos_sig_metric.summarize("average"),
                    "dnsmos_bak": 5* self.dnsmos_bak_metric.summarize("average"),
                    "dnsmos_ovr": 5* self.dnsmos_ovr_metric.summarize("average")


                }
                self.hparams.tensorboard_train_logger.log_stats(valid_stats)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=[self.hparams.target_metric]
            )

        if stage == sb.Stage.TEST:
            if self.supervised:
                self.hparams.train_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stats,
                )
            else:
                self.hparams.train_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter_stage2.current},
                    test_stats=stats,
                )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        "Override dataloader to insert custom sampler/dataset"
        if stage == sb.Stage.TRAIN:

            # Create a new dataset each time, this set grows
            if self.sub_stage == SubStage.HISTORICAL:
                dataset = sb.dataio.dataset.DynamicItemDataset(
                    data=dataset,
                    dynamic_items=[enh_pipeline],
                    output_keys=["id", "enh_sig", "score"],
                )
                samples = round(len(dataset) * self.hparams.history_portion)
            else:
                samples = self.hparams.number_of_samples

            # This sampler should give the same samples for D and G
            if self.supervised:
            
                epoch = self.hparams.epoch_counter.current
            else:
                epoch = self.hparams.epoch_counter_stage2.current
            #print(epoch)
            # Equal weights for all samples, we use "Weighted" so we can do
            # both "replacement=False" and a set number of samples, reproducibly
            weights = torch.ones(len(dataset))
            #print("making sampler!")
            sampler = ReproducibleWeightedRandomSampler(
                weights, epoch=epoch, replacement=False, num_samples=samples
            )
            sampler.set_epoch(epoch)
            loader_kwargs["sampler"] = sampler

            if self.sub_stage == SubStage.GENERATOR:
                self.train_sampler = sampler

        # Make the dataloader as normal
        return super().make_dataloader(
            dataset, stage, ckpt_prefix, **loader_kwargs
        )

    def on_fit_start(self):
        
        "Override to prevent this from running for D training"
        if self.sub_stage == SubStage.GENERATOR:
            super().on_fit_start()

    def init_optimizers(self):

        torchinfo.summary(self.modules.generator)
        print("______")
        torchinfo.summary(self.modules.degenerator)
        print("______")
        torchinfo.summary(self.modules.discriminator)
        print("______")
    
        "Initializes the generator and discriminator optimizers"
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )

        self.gd_optimizer = self.hparams.gd_opt_class(
            self.modules.degenerator.parameters()
        )

        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("g_opt", self.g_optimizer)
            self.checkpointer.add_recoverable("gd_opt", self.gd_optimizer)
            self.checkpointer.add_recoverable("d_opt", self.d_optimizer)


# Define audio pipelines
@sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
@sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
def audio_pipeline(noisy_wav, clean_wav):
    yield sb.dataio.dataio.read_audio(noisy_wav)
    yield sb.dataio.dataio.read_audio(clean_wav)


# For historical data
@sb.utils.data_pipeline.takes("enh_wav")
@sb.utils.data_pipeline.provides("enh_sig")
def enh_pipeline(enh_wav):
    yield sb.dataio.dataio.read_audio(enh_wav)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class."""

    # Define datasets
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(hparams_file, "run_opts\n",run_opts,"overrides:\n", overrides)
    if "local-rank" in overrides:
        import re
        rank = re.findall("local-rank: \d+", overrides)[0]
        overrides = overrides.replace(rank, "")
        rank = rank.split(":")[-1].strip()
        run_opts["local_rank"] = int(rank)
        print(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    #Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = MetricGanBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    

    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_folder"]})
    #------------------SUPERVISED-----------------

    from utils.dataset_setup import supervised_setup  # noqa
    args = {
    "train": ["libri1to3mix"],
    "val": ["libri1to3mix",],
    "test": ["libri1to3mix","reverb_libri1to3mix"],
    "batch_size": hparams["train_N_batch"],
    "val_batch_size": hparams["valid_N_batch"],
    "number_of_samples": hparams["number_of_samples"],
    "n_jobs": 1,
    "p_single_speaker": 1.0,
    "fs": 16000,
    "min_or_max": "max",
    "audio_timelength": 4.0,
    "use_vad": False,
    }
    datasets = supervised_setup(args)
    for d in datasets:
        print(d,type(datasets[d]),len(datasets[d]))

    se_brain.train_set = datasets["train"]
    se_brain.historical_set = {}

    se_brain.noisy_scores = {}
    se_brain.load_history()

    se_brain.batch_size = hparams["dataloader_options"]["batch_size"]
    se_brain.sub_stage = SubStage.GENERATOR
    se_brain.modules.teacher = None
    se_brain.supervised = True
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["val_libri1to3mix_1sp"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )




#------------------SEMI-SUPERVISED-----------------
    args2 = {
    "train": ["chime"],
    "val": ["libri1to3mix","libri1to3chime"],
    "test": ["chime"],
    "batch_size": hparams["train_N_batch"],
    "val_batch_size": hparams["valid_N_batch"],
    "n_jobs": 1,
    "number_of_samples": hparams["number_of_samples"],
    #"p_single_speaker": 0.5,
    "p_single_speaker": 1.0,
    "fs": 16000,
    "min_or_max": "min",
    "audio_timelength": 4.0,
    "use_vad": False,
    }
    datasets_stage2 = unsupervised_setup(args2)
    # print(datasets_stage2)
    for d in datasets_stage2:
        print(d,type(datasets_stage2[d]),len(datasets_stage2[d]))


    se_brain.train_set = datasets_stage2['train']
    
    
    import copy
  
    #set up the generator of the supervised training to be
    #the teacher of the unsupervised
    se_brain.modules.teacher = copy.deepcopy(se_brain.modules.generator.eval())
    assert se_brain.modules.teacher != se_brain.modules.generator
    se_brain.supervised = False
    #clear the historical set 
    se_brain.historical_set = {}
    print("starting unsupervised training!")
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter_stage2,
        train_set=datasets_stage2["train"],
        valid_set=datasets["val_libri1to3mix_1sp"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )
    se_brain.supervised = True
    test_stats = se_brain.evaluate(
        test_set=datasets["test_reverb_libri1to3mix_1sp"],
        max_key=hparams["target_metric"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
