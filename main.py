import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utils.utility import parse_args
from collections import Counter
from utils.SASRecModules_ori import *

from models.Caser import Caser
from models.Gru import GRU
from models.SASRec import SASRec
from models.BERT4Rec import BERT4Rec
from models.SHAN import SHAN

from train import train

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':

    args = parse_args()
    print("model name: %s" % args.model_name)
    print("loss_type: %s" % args.loss_type)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    # logging.basicConfig(filename="./log/{}/{}_{}_lr{}_decay{}_dro{}_gamma{}_beta{}".format(args.data + '_final2', Time.strftime("%m-%d %H:%M:%S", Time.localtime()), args.model_name, args.lr, args.l2_decay, args.dro_reg, args.gamma, args.beta)) #from bce
    # logging.basicConfig(filename="./log/{}/{}_{}_lr{}_decay{}_dro{}_gamma{}".format(args.data + '_final2', Time.strftime("%m-%d %H:%M:%S", Time.localtime()), args.model_name, args.lr, args.l2_decay, args.dro_reg, args.gamma)) #from bpr,mse
    # Network parameters

    data_directory = './data/' + args.data
    # data_directory = './data/' + args.data
    # data_directory = '../' + args.data + '/data'
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    if args.model_name == 'SHAN':
        user_num = 1
        data_statis['user_id'] = 1
    #print(data_statis.columns)
    print('seq_size, item_num=%s,%s' % (seq_size,item_num))
    reward_click = args.r_click
    reward_buy = args.r_buy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_name == 'Caser':
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters_h, args.num_filters_v, args.filter_sizes, args.dropout_rate)
    elif args.model_name == 'Gru':
        model = GRU(args.hidden_factor,item_num, seq_size)
    elif args.model_name == 'SASRec':
        model = SASRec(args.hidden_factor,item_num, seq_size, args.dropout_rate, device)
    elif args.model_name == 'BERT4Rec':
        model = BERT4Rec(args.hidden_factor,item_num, seq_size, args.dropout_rate,device)
    elif args.model_name == 'SHAN':
        model = SHAN(args.hidden_factor, item_num, user_num, seq_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # optimizer.to(device)
    
    train(args, item_num, optimizer, device, model, data_directory, seq_size,reward_click)
 
