# MultiCMGAN+/+
Code for the paper:
"MULTI-CMGAN+/+: LEVERAGING SPEECH QUALITY METRIC PREDICTION FOR
SPEECH ENHANCEMENT TRAINED ON REFERENCE-FREE REAL-WORLD DATA"
by George Close, William Ravenscroft, Thomas Hain, and Stefan Goetze

## Data
Uses data format and dataloading from [CHiME-7 UDASE task]().
See that task for data preperation guide. 
## Setup
Set variables in `__config__.py` to point to required training data

For the HuBERT representation, edit `HuBERT_wrapper.py` to point to the file `hubert_base_ls960.pt`

Conda environment for training is `chime.yaml`

## Training
To train the framework

```python3 train.py hparams/hyperparams_chime_bak_ovr_pesq_1.0.yaml```


or use one of the other provided hyperparameter files. 

## Evaluation
Use the `eval_cmgan.py` script to evaluate a trained model. See the command line arguments for this script for details. 