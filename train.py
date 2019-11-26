import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
import torch
import random
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')

parser.add_argument('--data_path', type = str, default = './prepro_data')
parser.add_argument('--checkpoint_dir', type = str, default = './checkpoint')
parser.add_argument('--fig_result_dir', type = str, default = './fig_result')


args = parser.parse_args()

model = {
    'CNN3': models.CNN3,
    'LSTM': models.LSTM,
    'BiLSTM': models.BiLSTM,
    'ContextAware': models.ContextAware,
    'OriBiLSTM': models.OriBiLSTM,
}

seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

con = config.Config(args)
con.set_max_epoch(200)
con.set_data_path(args.data_path)
con.set_checkpoint_dir(args.checkpoint_dir)
con.set_fig_result_dir(args.fig_result_dir)
print("data path: ", con.data_path)
print("checkpoint dir: ", con.checkpoint_dir)
print("fig result dir: ", con.fig_result_dir)

con.load_train_data()
con.load_test_data()
# con.set_train_model()

con.train(model[args.model_name], args.save_name)

def check_args(args):
    print('------------ Options -------------')
    for k in args.__dict__:
        v = args.__dict__[k]
        print('%s: %s' % (str(k), str(v)))
    print('------------ End -------------')
