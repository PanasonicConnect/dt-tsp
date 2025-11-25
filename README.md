# Offline Decision Transformers for Neural Combinatorial Optimization: Surpassing Heuristics on the Traveling Salesman Problem

This repository implements the paper [Offline Decision Transformers for Neural Combinatorial Optimization: Surpassing Heuristics on the Traveling Salesman Problem](https://openreview.net/forum?id=TLLSQitTo3), presented at the [NeurIPS 2025 Differentiable Learning of Combinatorial Algorithms Workshop](https://sites.google.com/view/diffcoalg2025).

## Installation

```bash
conda env create --file environment.yml
```

## Data preparation

Place all datasets in the `data/` directory.

1. Download or generate the TSP instance data from [learning-paradigms-for-tsp](https://github.com/chaitjo/learning-paradigms-for-tsp).
2. Then, generate the corresponding solution data.

## Training

Train a model on Farthest Insertion (FI) for N=20:

```bash
python train.py \
    --train_dataset=data/tsp20_train_fi.txt \
    --val_dataset=data/tsp20_val_fi.txt \
    --log_dir=dt_runs/dt_n20_fi \
    --num_epochs=2000 \
    --save_iters=10
```

## Evaluation

Evaluate the trained model:

```bash
python eval.py \
    --test_dataset=data/tsp20_test_fi.txt \
    --model_dir=dt_runs/dt_n20_fi_[%y%m%d%H%M%S] \
    --chk_pt_name="best.pt" \
    --eval_optimal_gap \
    --optimal_dataset=data/tsp20_test_concorde.txt
```

## Acknowledgement

This implementation is based on [Elastic-DT](https://github.com/kristery/Elastic-DT) and [learning-paradigms-for-tsp](https://github.com/chaitjo/learning-paradigms-for-tsp).