import hashlib
import random

import numpy as np

clust_script = "/home/yalavarthi/seperable_flows/sbatch_file.sh"
run_hp = open("run_hp.sh", "w")
ct = 36
ft = 0
for dataset in ["ushcn", "physionet2012", "mimiciii", "mimiciv"]:
    fold = 0
    basefolder = "/home/yalavarthi/seperable_flows"
    for i in range(710, 720):
        n_gaussians = random.choice([1, 3, 5, 7, 10])
        f_layers = random.choice([3])
        n_heads = random.choice([1, 2, 4])
        n_hidden = random.choice([32, 64, 128])
        enc_layers = random.choice([2, 3, 4])
        batch = f"sbatch --job-name={dataset} --output={basefolder}/results/{dataset}/{dataset}-mnf-ct-{ct}-ft-{ft}-{i}-%A.log --error={basefolder}/results/{dataset}/{dataset}-mnf-ct-{ct}-ft-{ft}-{i}-%A.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de {clust_script} train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset {dataset} --n-gaussians {n_gaussians} --flayers {f_layers} --n-heads {n_heads} --encoder grafiti --enc-layers {enc_layers} --n-hidden {n_hidden} -ct {ct} -ft {ft} --fold {fold} \n"
        run_hp.write(batch)
run_hp.close()
