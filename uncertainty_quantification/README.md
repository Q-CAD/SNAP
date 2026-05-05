# Per-atom Uncertainty Quantification for Machine Learning Interatomic Potentials (MLIPs)

This repository contains code for training and evaluating the Uncertainty Quantification for Machine Learning Interatomic Potentials (MLIPs) method described in the paper "Per-atom Uncertainty Quantification for Machine Learning Interatomic Potentials (MLIPs)". The code is organized into several scripts that can be used to train the MLIP model, generate embeddings, and evaluate the model's performance.

## Requirements

The code requires Python 3.8 or higher, your MLIP package of choice (such as [MACE](https://github.com/ACEsuit/mace) or [UMA](https://huggingface.co/facebook/UMA)), and the following packages:
- numpy
- pandas
- ase
- torch
- xgboost


## Usage

### Extract per-atom embeddings and energies using a trained MLIP

To train the GBM model, we first need to extract per-atom embeddings and per-atom energies from a trained MLIP. The following commands provide methods to extract this information for MACE and UMA. If using a finetuned checkpoint, the `--checkpoint` flag can be used to specify the path to the checkpoint file. The sample should be in a format readable by ASE and contain configurations *from the validation set* used to train or finetune the MLIP.

```
python run_embeddings_mace.py \
  --sample data/example.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

```
python run_embeddings_uma.py \
  --sample data/example.xyz \
  --savedir data/embeddings_uma \
  --model-size uma-s-1p1 \
  --head 'omat' \
  --index ":"
```


### Train GBM on per-atom embeddings and energies

Once the embeddings and energies have been extracted,the GBM model can be trained using the following command. 

```
python train-gbm.py --embeddings data/embeddings_mace/embedding_info_example.npz \
	--savedir data/gbm_mace \
  --upper-alpha 0.95 \
  --lower-alpha 0.05 \
  --estimators 1000 
  ```


### Compute per-atom uncertainties using the trained GBM model

To compute per-atom uncertainties for a trajectory produced using the MLIP, the per-atom embeddings must be extracted in the same was as described above. 

```
python run_embeddings_mace.py \
  --sample data/md_run.xyz \
  --savedir data/embeddings_mace \
  --model-size medium-0b \
  --index ":"
```

Then, the following command can be used to compute per-atom uncertainties using the trained GBM model.

```
python run-gbm.py --embeddings 'data/embeddings/embedding_info_md_run.npz' --savedir 'results/gbm_mace'
```

### ADDITION: Compute normalized total energy uncertainties for a collection of structures
quantile_prediction -h

### ADDITION: Select the most uncertain images from a pool of different structure predictions
structure_selection -h

### ADDITION: Create cubic simulation cells from the sites with the largest range of energy contributions based on GBM regressors
screen_sites -h

## Citation
If you use this code in your research, please cite the following paper:

