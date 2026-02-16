# seperable_flows

This is the source code for the paper [Reliable Probabilistic Forecasting of Irregular Time Series through Marginalization-Consistent Flows](https://openreview.net/forum?id=awWi4hJI7O)

# Requirements
Please use moses.yaml for requirements

# Training and Evaluation
```
train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset ushcn --n-gaussians 7 --flayers 3 --n-heads 2 --n-hidden 64 -ct 36 -ft 0 --fold 0
train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset physionet2012 --n-gaussians 10 --flayers 3 --n-heads 1 --n-hidden 64 -ct 36 -ft 0 --fold 0
train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset mimiciii --n-gaussians 3 --flayers 3 --n-heads 1 --n-hidden 64 -ct 36 -ft 0 --fold 0
train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset mimiciv --n-gaussians 3 --flayers 3 --n-heads 1 --n-hidden 128 -ct 36 -ft 0 --fold 0
```

To download MIMIC-III and MIMIC-IV, a permission is required. Once, the datasets are downloaded, add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. We use the TSDM from [https://github.com/yalavarthivk/GraFITi/].