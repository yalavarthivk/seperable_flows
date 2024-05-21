import argparse
import os
import pdb
import random
import sys
import time
from random import SystemRandom

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mnf import load_data, utils
from mnf.MNF import MarginalNormalizingFlows

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
# fmt: off
# pylint: disable=consider-using-f-string

parser = argparse.ArgumentParser(description="Training Script for ProFITi.")
parser.add_argument("-e",  "--epochs",       default=1000,    type=int,   help="maximum epochs")
parser.add_argument("-bs", "--batch-size",   default=64,     type=int,   help="batch-size")
parser.add_argument("-vbs", "--val-batch-size", default=50,     type=int,   help="val-batch-size")
parser.add_argument("-lr", "--learn-rate",   default=0.001,  type=float, help="learn-rate")
parser.add_argument("-b",  "--betas", default=(0.9, 0.999),  type=float, help="adam betas", nargs=2)
parser.add_argument("-wd", "--weight-decay", default=0.001,  type=float, help="weight-decay")
parser.add_argument("-s",  "--seed",         default=None,   type=int,   help="Set the random seed.")
parser.add_argument("-dset", "--dataset", default="ushcn", type=str, help="Name of the dataset")
parser.add_argument("-fl", "--flayers", default=1, type=int, help="number of layers in the flow")
parser.add_argument("-ng", "--n-gaussians", default=3, type=int, help="number of Gaussian components")
parser.add_argument("--use-cov", default=1, type=int, help="should we use covariance matrix for the base gaussian, 1 for True and 0 for False")
parser.add_argument("--use-activation", default=0, type=int, help="should we use activation function between splines, 1 for True and 0 for False")
parser.add_argument("-nh", "--n-heads", default=4, type=int, help="number of heads for mha in encoder")
parser.add_argument("--n-hiddens", default=128, type=int, help="#number of hidden dimensions")
parser.add_argument("--patience", default=30, type=int, help="patience for early stopping" )
parser.add_argument("-ft", "--forc-time", default=0, type=int, help="forecast horizon in hours")
parser.add_argument("-ct", "--cond-time", default=36, type=int, help="conditioning range in hours")
parser.add_argument("-nf", "--nfolds", default=5, type=int, help="#folds for crossvalidation")
parser.add_argument("--fold", default=0, type=int, help="#fold in crossvalidation")
# fmt: on
ARGS = parser.parse_args()
print(" ".join(sys.argv))
experiment_id = int(SystemRandom().random() * 10000000)
print(ARGS, experiment_id)

if ARGS.seed is not None:
    torch.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)

OPTIMIZER_CONFIG = {
    "lr": ARGS.learn_rate,
    "betas": torch.tensor(ARGS.betas),
    "weight_decay": ARGS.weight_decay,
}

if ARGS.dataset == "ushcn":
    from tsdm.tasks import USHCN_DeBrouwer2019

    TASK = USHCN_DeBrouwer2019(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "mimiciii":
    from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

    TASK = MIMIC_III_DeBrouwer2019(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "mimiciv":
    from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021

    TASK = MIMIC_IV_Bilos2021(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )
elif ARGS.dataset == "physionet2012":
    from tsdm.tasks.physionet2012 import Physionet2012

    TASK = Physionet2012(
        normalize_time=True,
        condition_time=ARGS.cond_time,
        forecast_horizon=ARGS.forc_time,
        num_folds=ARGS.nfolds,
    )

TRAIN_LOADER, VAL_LOADER, TEST_LOADER = load_data.data_loaders(TASK, ARGS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### logging

LOGGING_DIR = "tensorboard/" + "log_dir-" + str(experiment_id)
os.makedirs(LOGGING_DIR, exist_ok=True)
WRITER = SummaryWriter(LOGGING_DIR)

## Checkpointing

chkpoint_path = "saved_models/mnf" + str(experiment_id) + ".h5"


MODEL_CONFIG = {
    "n_inputs": TASK.dataset.shape[-1],
    "n_gaussians": ARGS.n_gaussians,
    "use_cov": ARGS.use_cov,
    "use_activation": ARGS.use_activation,
    "n_hiddens": ARGS.n_hiddens,
    "n_flayers": ARGS.flayers,
    "n_heads": ARGS.n_heads,
    "device": DEVICE,
}


MODEL = MarginalNormalizingFlows(**MODEL_CONFIG).to(DEVICE)
MODEL.zero_grad(set_to_none=True)

# ## Initialize Optimizer
from torch.optim import AdamW

compute_loss = utils.compute_losses()

OPTIMIZER = AdamW(MODEL.parameters(), **OPTIMIZER_CONFIG)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER, "min", patience=10, factor=0.5, min_lr=0.00001, verbose=True
)
nepochs = ARGS.epochs
BEST_VAL_LOSS = 1e8
ES = False
EARLY_STOP = 0

for epoch in range(0, nepochs):
    TRAIN_NLL = 0
    COUNT = 0
    MODEL.train()
    ts = time.time()
    for batch in TRAIN_LOADER:
        OPTIMIZER.zero_grad()
        OBS, MOBS, XQ, MQ, Y = (tensor.to(DEVICE) for tensor in batch)
        if XQ.shape[1]==0:
            continue
        Z, LDJ, mW = MODEL(OBS, MOBS, XQ, MQ, Y)
        nsamples = MQ.sum(-1).bool().sum()
        loss = compute_loss.nll(Z, MQ, LDJ, mW)
        TRAIN_NLL += loss * nsamples
        COUNT += nsamples
        loss.backward()
        OPTIMIZER.step()
    te = time.time()
    print(
        "epoch: {}, train_loss: {:.6f}, epoch_time: {:.2f}".format(
            epoch, TRAIN_NLL / COUNT, te - ts
        )
    )
    WRITER.add_scalar("Loss/train_nll", TRAIN_NLL / COUNT, epoch)  # write train nl loss
    WRITER.add_scalar("time-per-epoch", te - ts, epoch)  # write run time

    MODEL.eval()
    with torch.no_grad():
        VAL_NLL = 0.0
        VAL_COUNT = 0.0
        ts = time.time()
        for batch in VAL_LOADER:
            OBS, MOBS, XQ, MQ, Y = (tensor.to(DEVICE) for tensor in batch)
            if XQ.shape[1]==0:
                continue
            nsamples = MQ.sum(-1).bool().sum()
            Z, LDJ, mW = MODEL(OBS, MOBS, XQ, MQ, Y)
            loss = compute_loss.nll(Z, MQ, LDJ, mW)
            VAL_NLL += loss * nsamples
            VAL_COUNT += nsamples
        te = time.time()
        VAL_NLL = VAL_NLL / VAL_COUNT
        print(
            "val_loss: {:.6f}, val_time: {:.2f}".format(
                VAL_NLL,
                te - ts,
            )
        )
    WRITER.add_scalar("Loss/val", VAL_NLL / VAL_COUNT, epoch)  # write val loss
    WRITER.add_scalar("val-time-per-epoch", te - ts, epoch)  # write val infer time

    if BEST_VAL_LOSS > VAL_NLL:
        BEST_VAL_LOSS = VAL_NLL
        torch.save(
            {
                "epoch": epoch,
                "state_dict": MODEL.state_dict(),
                "optimizer_state_dict": OPTIMIZER.state_dict(),
                "loss": VAL_NLL,
            },
            chkpoint_path,
        )
        EARLY_STOP = 0
    else:
        EARLY_STOP += 1

    scheduler.step(VAL_NLL)
    if (EARLY_STOP == ARGS.patience) or (epoch == nepochs - 1):
        if EARLY_STOP == 30:
            print(
                "Early stopping because of no improvement in val. metric for 30 epochs"
            )
        else:
            print("Completed all the epochs")

        chp = torch.load(chkpoint_path)  # load the checkpoint
        MODEL.load_state_dict(chp["state_dict"])

        with torch.no_grad():
            TEST_NLL = 0.0
            TEST_COUNT = 0.0
            ts = time.time()
            for batch in TEST_LOADER:
                OBS, MOBS, XQ, MQ, Y = (tensor.to(DEVICE) for tensor in batch)
                if XQ.shape[1]==0:
                    continue
                Z, LDJ, mW = MODEL(OBS, MOBS, XQ, MQ, Y)
                nsamples = MQ.sum(-1).bool().sum()
                loss = compute_loss.nll(Z, MQ, LDJ, mW)
                TEST_NLL += loss * nsamples
                TEST_COUNT += nsamples
            te = time.time()
            TEST_NLL = TEST_NLL / TEST_COUNT
            print(
                "best_val_loss: {:.6f}, test_loss: {:.6f}, test_time: {:.2f}".format(
                    BEST_VAL_LOSS,
                    TEST_NLL,
                    te - ts,
                )
            )

            WRITER.add_scalar(
                "test loss", TEST_NLL / TEST_COUNT, epoch
            )  # write test loss
            WRITER.add_scalar("test time", te - ts, epoch)  # write test infer time

        break
