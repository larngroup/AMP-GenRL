# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 09:50:45 2022

@author: Tiago
"""

import argparse
import os


def argparser():
    """
    Argument Parser Function

    Outputs:
    - FLAGS: arguments object

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--option',
        type=str,
        default='optimization',
        help='pre_train, fine_tune, evaluation, optimization or validation')

    parser.add_argument(
        '--data_path',
        type=str,
        default='data//example_data.txt',
        help='Data Path')

    parser.add_argument(
        '--test_rate',
        type=float,
        default=0.1,
        help='Rate of test set')

    parser.add_argument(
        '--subset_instances',
        type=int,
        default=100,
        help='Number of instances used to estimate protein properties from the training set')

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=25,
        help='Number of epochs')

    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='WarmUpDefault',
        help='WarmUpDefault or SGDR')

    parser.add_argument(
        '--lr_WarmUpSteps',
        type=int,
        default=8000,
        help="Warmup steps - only for WarmUpDefault")

    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action ='append',
        help='Optimizer Function Parameters - optimizer, learning rate, beta1, beta2 and epsilon')

    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.001,
        help='Minimum delta')

    parser.add_argument(
        '--patience',
        type=int,
        default=15,
        help='Epochs to keep training without improvement')

    parser.add_argument(
        '--d_model',
        type=int,
        nargs='+',
        help='Model Dimension')

    parser.add_argument(
        '--n_layers',
        type=int,
        nargs='+',
        help='Number of layers')

    parser.add_argument(
        '--n_heads',
        type=int,
        nargs='+',
        help='Number of heads')

    parser.add_argument(
        '--dropout',
        type=float,
        nargs='+',
        help='Dropout rate')

    parser.add_argument(
        '--activation_func',
        type=str,
        nargs='+',
        help='Feed-forward activation function')

    parser.add_argument(
        '--ff_dim',
        type=int,
        nargs='+',
        help='Feed-forward Dimension')

    parser.add_argument(
        '--batchsize',
        type=int,
        default=8,
        help='Batch size')

    parser.add_argument(
        '--max_strlen_pt',
        type=int,
        default=200,
        help='Maximum string length')
    
    parser.add_argument(
        '--max_strlen_ft',
        type=int,
        default=200,
        help='Maximum string length')
    
    parser.add_argument(
        '--n_generate',
        type=int,
        default=250,
        help='Number of proteins to generate')
    
    parser.add_argument(
        '--save_peptides',
        type=bool,
        default=False,
        help='Save best sampled peptides')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='Directory for checkpoint weights'
    )

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS


def logging(msg, FLAGS):
    """
    Logging function to update the log file

    Args:
    - msg [str]: info to add to the log file
    - FLAGS: arguments object

    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")

    with open(fpath, "a") as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")