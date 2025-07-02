# Parametric Scaling Law of Tuning Bias in Conformal Prediction

This repository is the official implementation of [Parametric Scaling Law of Tuning Bias in Conformal Prediction](https://openreview.net/forum?id=jnJLZXSOin) at ICML 2025. 

## Overview
Conformal prediction is a popular framework of uncertainty quantification that constructs prediction sets with coverage guarantees. To uphold the exchangeability assumption, many conformal prediction methods necessitate an additional hold-out set for parameter tuning. Yet, the impact of violating this principle on coverage remains underexplored, making it ambiguous in practical applications. In this work, we empirically find that the tuning bias - the coverage gap introduced by leveraging the same dataset for tuning and calibration, is negligible for simple parameter tuning in many conformal prediction methods.  In particular, we observe the scaling law of the tuning bias: this bias increases with parameter space complexity and decreases with calibration set size. Formally, we establish a theoretical framework to quantify the tuning bias and provide rigorous proof for the scaling law of the tuning bias by deriving its upper bound. In the end, we discuss how to reduce the tuning bias, guided by the theories we developed.

## How to Install

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

This project uses the ImageNet dataset as default. You need to modify the dataset path in `data/dataset_imagenet.py`:

1. Open `data/dataset_imagenet.py`
2. Locate the following line:
   ```python
   validir = os.path.join(data_dir, '/mnt/sharedata/ssd3/common/datasets/imagenet/images/val')
   ```
3. Replace `/mnt/sharedata/ssd3/common/datasets/imagenet/images/val` with your own ImageNet validation directory path

Additionally, ensure that you update any other data paths if needed for your specific environment.

## How to Run

This repository contains three experiments in Section Empirical Study of our paper. The settting of each experiment is shown in our paper.

### 1. Confidence Calibration Experiment

```bash
python exp_confidence_calibration.py \
    --cal_num 5000 \
    --num_classes 1000 \
    --conformal thr \
    --alpha 0.1 \
    --num_runs 30 \
    --freeze_num 0 \
    --preprocess vs \
    --file results/confidence_calibration_results.txt \
    --device cuda:0
```

### 2. Score Parameter Tuning Experiment

```bash
python exp_score_parameter.py \
    --cal_num 5000 \
    --num_classes 1000 \
    --conformal raps \
    --alpha 0.1 \
    --num_runs 20 \
    --file results/score_parameter_results.txt \
    --preprocess ts \
    --device cuda:0
```

### 3. Score Weight Experiment

```bash
python exp_score_weight.py \
    --cal_num 5000 \
    --num_classes 1000 \
    --alpha 0.1 \
    --num_runs 20 \
    --file results/score_weight_results.txt \
    --preprocess ts \
    --device cuda:0
```

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@inproceedings{zeng2025parametric,
  title={Parametric Scaling Law of Tuning Bias in Conformal Prediction},
  author={Hao Zeng and Kangdao Liu and Bingyi Jing and Hongxin Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
``` 