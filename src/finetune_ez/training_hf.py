from argparse import ArgumentParser
from typing import Union
import os
import logging
import json
import pandas as pd
import glob

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
import torch
from datasets import load_from_disk, Dataset

from finetune_ez import utils


def load_hf_model(str_d: str) -> AutoModelForSequenceClassification:
    pass


def load_hf_tokenizer(str_d: str) -> Union[AutoTokenizer, None]:
    if str_d:
        return AutoTokenizer.from_pretrained(str_d)
    else:
        return None
    
OPTIM_CHOICES = {
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'nadam': torch.optim.NAdam,
}


def load_optimizer(optim: str):
    assert (optim in OPTIM_CHOICES)
    return OPTIM_CHOICES[optim]



def add_common_args(parser: ArgumentParser):
    parser.add_argument('--tokenizer', action='store', type=load_hf_tokenizer, default=None,
                        help='name or dir for HF tokenizer')
    
    parser.add_argument('--model', action='store', type=str, required=True,
                        help='name of the model, modelpath from HF')
    
    parser.add_argument('--output-dir', action='store', required=True, help='path to output')
    parser.add_argument('--num-labels', action='store', type=int, default=2, help='nums of labels')
    parser.add_argument('--problem-type', action='store', type=str, default='single_label_classification', 
                        choices=['regression', 'single_label_classification', 'multi_label_classification']
                                + utils.ALTERNATIVE_LOSSES,
                                help='problem type under which you want to train the model'
                                )
    
    parser.add_argument('--lr', action='store', type=float, default=2e-5, help='training learning rate')
    parser.add_argument('--optimizer', action='store', type=load_optimizer, default='adamw')
    parser.add_argument('--validation-data', action='store', type=load_from_disk, required=True,
                        help='validation data directory')
    
    parser.add_argument('--test-data', action='store', type=load_from_disk, required=False,
                        help='test data directory')
    parser.add_argument('--lr-plateau-patience', action='store', type=int, default=None,
                        help='once the patience is reached, earlystoppinf is triggered to stop the training based on eval_f1 as best metirc')
    
    parser.add_argument('--lr-plateau-factor', action='store', type=float, default=0.1,
                        help='factor by which the learning rate will be reduced')
    
    parser.add_argument('--batch-size', action='store', type=int, required=True, help='batch size')

def add_training_args(parser: ArgumentParser):
    parser.add_argument('--labeled-data', action='store', type=load_from_disk, required=True,
                        help='training data')
    #
    parser.add_argument('--max-training-epochs', action='store', type=int, default=32, required=False,
                        help='maximum epochs to spend retraining model')

    parser.add_argument('--save-initialization', action='store_true', default=False, required=False,
                        help='save the random initialization')
    parser.add_argument('--reweight', action='store_true', default=False, required=False,
                                help='Reweight')
    



def _run_train_model_with_hf():
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser('train-with-hf')
    add_common_args(parser)
    add_training_args(parser)
    args = parser.parse_args()


    model = utils.run_train_model_with_hf(args.model,args.labeled_data, args.validation_data,
                                                      args.optimizer, args.output_dir, batch_size=args.batch_size,
                                                      n_epochs=args.max_training_epochs, lr=args.lr,
                                                  lr_plateau_patience=args.lr_plateau_patience,
                                                  lr_plateau_factor=args.lr_plateau_factor, tokenizer=args.tokenizer,
                                                  save_initialization=args.save_initialization,
                                                  problem_type=args.problem_type, num_labels=args.num_labels,
                                                  reweight=args.reweight)

    model.save_pretrained(os.path.join(args.output_dir, 'trained-model'))