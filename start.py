# Add all configs and startup code here. Maybe a good idea if:
# 1. Parse configs and other setup code (probably what OpenNIR wants)
# 2. Init Trainer / OpenNIR pipeline with configs
# 3. From there everything should be encapsulated in the Trainer OpenNIR pipeline

import argparse

import torch
import numpy as np

import onir


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


if __name__ == '__main__':
    main()

    # parser = argparse.ArgumentParser(description="General setup")
    #
    # # General
    # parser.add_argument('--exp_name', type=str, default="", help="Name of the experiment for house-keeping stuff")
    # parser.add_argument('--run-id', type=str, default="", help="Arbitrary run-id to further distinguish experiments")
    # parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    # parser.add_argument('--config', default="", type=str, help='Path to the config file used')
    # parser.add_argument('--seed', type=int, default=42, help="Seed that is used")
    #
    # ARGS = parser.parse_args()
    #
    # torch.manual_seed(ARGS.seed)
    # np.random.seed(ARGS.seed)
