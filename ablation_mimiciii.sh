#!/bin/bash

# MIMIC-III - varying number of mixtures

for FOLD in {0..4}; do
    for N_GAUSSIAN in 1 2 3 5 7; do
        sbatch --job-name=mimiciii-abl-d --output=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_n_gaussian_${N_GAUSSIAN}_fold_${FOLD}_51${FOLD}.log --error=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_n_gaussian_${N_GAUSSIAN}_fold_${FOLD}_51${FOLD}.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de /home/yalavarthi/seperable_flows/sbatch_file.sh train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset mimiciii --n-gaussians ${N_GAUSSIAN} --encoder transformer --flayers 3 --n-heads 1 --n-hidden 64 --fold ${FOLD}
    done
done

# MIMIC-III - varying latent size

for FOLD in {0..4}; do
    for LATENT_SIZE in 16 32 64 128; do
        sbatch --job-name=mimiciii-abl-m --output=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_latent_size_${LATENT_SIZE}_fold_${FOLD}_51${FOLD}.log --error=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_latent_size_${LATENT_SIZE}_fold_${FOLD}_51${FOLD}.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de /home/yalavarthi/seperable_flows/sbatch_file.sh train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset mimiciii --n-gaussians 3 --encoder transformer --flayers 3 --n-heads 1 --n-hidden ${LATENT_SIZE} --fold ${FOLD}
    done
done

# MIMIC-III - varying number of flow layers

for FOLD in {0..4}; do
    for FLAYERS in 0 1 2 3; do
        sbatch --job-name=mimiciii-abl-fl --output=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_flayers_${FLAYERS}_fold_${FOLD}_51${FOLD}.log --error=/home/yalavarthi/seperable_flows/ablation_results/mimiciii/abl_flayers_${FLAYERS}_fold_${FOLD}_51${FOLD}.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de /home/yalavarthi/seperable_flows/sbatch_file.sh train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset mimiciii --n-gaussians 3 --encoder transformer --flayers ${FLAYERS} --n-heads 1 --n-hidden 64 --fold ${FOLD}
    done
done