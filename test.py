import json
import torch
from sqlnet.utils import *
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    parser.add_argument('--cnn', action='store_true',
            help='Use cnn for predicting num of where clause')
    parser.add_argument('--filter_num', type=int, default=1,
            help='1: defulat filter size')
    parser.add_argument('--agg', action='store_true',
            help='include agg')
    parser.add_argument('--sel', action='store_true',
            help='include sel')
    parser.add_argument('--cond', action='store_true',
            help='include cond')
    args = parser.parse_args()


    if args.toy:
        USE_SMALL=True
        GPU=True
        N_word=100
        B_word=6
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        N_word=300
        B_word=42
        BATCH_SIZE=64
    TEST_ENTRY=(args.agg, args.sel, args.cond)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    args.dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=False, use_small=USE_SMALL) # load_used can speed up loading

    model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, use_cnn=args.cnn, filter_num=args.filter_num, gpu=GPU,
                trainable_emb = False, agg=args.agg, sel=args.sel, cond=args.cond)

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
        if args.agg:
            print ("Loading from %s"%agg_m)
            model.agg_pred.load_state_dict(torch.load(agg_m))
            print ("Loading from %s"%agg_e)
            model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        if args.sel:
            print ("Loading from %s"%sel_m)
            model.sel_pred.load_state_dict(torch.load(sel_m))
            print ("Loading from %s"%sel_e)
            model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        if args.cond:
            print ("Loading from %s"%cond_m)
            model.cond_pred.load_state_dict(torch.load(cond_m))
            print ("Loading from %s"%cond_e)
            model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(args)
        if args.agg:
            print ("Loading from %s"%agg_m)
            model.agg_pred.load_state_dict(torch.load(agg_m))
        if args.sel:
            print ("Loading from %s"%sel_m)
            model.sel_pred.load_state_dict(torch.load(sel_m))
        if args.cond:
            print ("Loading from %s"%cond_m)
            model.cond_pred.load_state_dict(torch.load(cond_m))

    print ("Dev err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s"%epoch_error(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY))
    print ("Dev acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY))
    print ("Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, TEST_ENTRY))
    print ("Test err num: %s;  breakdown on\n (agg, sel, where, agg_x_sel_o, agg_o_sel_x, cond_num, cond_col, cond_op, cond_val):\n %s"%epoch_error(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY))
    print ("Test acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY))
    print ("Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, TEST_ENTRY))
