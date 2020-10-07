import argparse

import torch
import numpy as np

import onir

import datasets, engine, modeling, utils


def main():
    context = onir.injector.load({
        'vocab': onir.vocab,
        'train_ds': onir.datasets,
        'ranker': onir.rankers,
        'trainer': onir.trainers,
        'valid_ds': onir.datasets,
        'valid_pred': onir.predictors,
        'test_ds': onir.datasets,
        'test_pred': onir.predictors,
        'pipeline': onir.pipelines,
    }, pretty=True)

    print(context)

    context['pipeline'].run()


if __name__ == "__main__":
    main()
