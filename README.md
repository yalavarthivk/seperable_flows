# seperable_flows

This is the source code for the paper ''Marginalization Consistent Mixture of Separable Flows''

# Requirements
python		3.8.11

Pytorch		1.9.0

sklearn		0.0

numpy		1.19.3


# Training and Evaluation

We provide an example using ''phsyionet``.

```
train_mnf.py --batch-size 64 --epochs 1000 -lr 0.001 --dataset physionet2012 --n-gaussians 10 --use-cov 1 --flayers 3 --n-heads 1 --n-hidden 64 -ct 36 -ft 0 --fold 0
```

To download MIMIC-III and MIMIC-IV, a permission is required. Once, the datasets are downloaded, add them to the folder .tsdm/rawdata/ and use the TSDM package to extract the folds. We use the TSDM from [https://github.com/yalavarthivk/GraFITi/].
