import json
import torch
from programs.utils import *
from programs.model.predictor import Predictor
import numpy as np
import datetime

import argparse
import os



def do_test(args):
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

    dataset = 1
    # 0: original dataset, 1: re-split dataset

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=False, use_small=USE_SMALL) # load_used can speed up loading

    model = Predictor(word_emb, N_word=N_word, use_ca=args.ca, use_cnn=args.cnn, use_col_cnn=args.col_cnn, filter_num=args.filter_num, cnn_type=args.cnn_type, gpu=GPU,
                trainable_emb = False, agg=args.agg, sel=args.sel, cond=args.cond, use_detach=args.detach)

    for i in range(100):
        if args.toy:
            if not os.path.isfile("result/toy_test%d.txt"%i):
                result_file = open("result/toy_test%d.txt"%i, 'w')
                break;
        else:
            if not os.path.isfile("result/real_test%d.txt"%i):
                result_file = open("result/real_test%d.txt"%i, 'w')
                break;

    args_dic = vars(args)
    for arg in args_dic:
        result_file.write("%s = %s\n"%(arg, args_dic[arg]))

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(dataset, args)
        if args.agg:
            print ("Loading from %s"%agg_m)
            result_file.write("Loading from %s\n"%agg_m)
            model.agg_pred.load_state_dict(torch.load(agg_m))
            print ("Loading from %s"%agg_e)
            result_file.write("Loading from %s\n"%agg_e)
            model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        if args.sel:
            print ("Loading from %s"%sel_m)
            result_file.write("Loading from %s\n"%sel_m)
            model.sel_pred.load_state_dict(torch.load(sel_m))
            print ("Loading from %s"%sel_e)
            result_file.write("Loading from %s\n"%sel_e)
            model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        if args.cond:
            print ("Loading from %s"%cond_m)
            result_file.write("Loading from %s\n"%cond_m)
            model.cond_pred.load_state_dict(torch.load(cond_m))
            print ("Loading from %s"%cond_e)
            result_file.write("Loading from %s\n"%cond_e)
            model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(dataset, args)
        if args.agg:
            print ("Loading from %s"%agg_m)
            result_file.write("Loading from %s\n"%agg_m)
            model.agg_pred.load_state_dict(torch.load(agg_m))
        if args.sel:
            print ("Loading from %s"%sel_m)
            result_file.write("Loading from %s\n"%sel_m)
            model.sel_pred.load_state_dict(torch.load(sel_m))
        if args.cond:
            print ("Loading from %s"%cond_m)
            result_file.write("Loading from %s\n"%cond_m)
            model.cond_pred.load_state_dict(torch.load(cond_m))

    val_err_num = epoch_error(model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY)
    print ("Dev err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s"%val_err_num)
    result_file.write("Dev err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n"%val_err_num)
    val_acc_qm = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY)
    print ("Dev acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s"%val_acc_qm)
    result_file.write("Dev acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s\n"%val_acc_qm)
    val_exec_acc = epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, TEST_ENTRY)
    print ("Dev execution acc: %s"%val_exec_acc)
    result_file.write("Dev execution acc: %s\n"%val_exec_acc)
    val_micro_acc = micro_cond_epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY)
    print ("Dev total cond num: %s\n accurate cond num breakdown on\n(cond_acc, cond_col, cond_op, cond_val):\n %s\n cond accuracy breakdowon on\n (cond_acc, cond_col, cond_op, cond_val):\n %s"%val_micro_acc)
    result_file.write("Dev total cond num: %s\n accurate cond num breakdown on\n(cond_acc, cond_col, cond_op, cond_val):\n %s\n cond accuracy breakdowon on\n (cond_acc, cond_col, cond_op, cond_val):\n %s\n"%val_micro_acc)
    test_err_num = epoch_error(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
    print ("Test err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s"%test_err_num)
    result_file.write("Test err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n"%test_err_num)
    test_acc_qm = epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
    print ("Test acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s"%test_acc_qm)
    result_file.write("Test acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val): %s\n"%test_acc_qm)
    test_exec_acc = epoch_exec_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, TEST_ENTRY)
    print ("Test execution acc: %s"%test_exec_acc)
    result_file.write("Test execution acc: %s\n"%test_exec_acc)
    test_micro_acc = micro_cond_epoch_acc(model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
    print ("Test total cond num: %s\n accurate cond num breakdown on\n(cond_acc, cond_col, cond_op, cond_val):\n %s\n cond accuracy breakdowon on\n (cond_acc, cond_col, cond_op, cond_val):\n %s"%test_micro_acc)
    result_file.write("Test total cond num: %s\n accurate cond num breakdown on\n(cond_acc, cond_col, cond_op, cond_val):\n %s\n cond accuracy breakdowon on\n (cond_acc, cond_col, cond_op, cond_val):\n %s\n"%test_micro_acc)
    result_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    parser.add_argument('--cnn', action='store_true',
            help='Use cnn for predicting num of where clause')
    parser.add_argument('--col_cnn', action='store_true',
            help='apply cnn to column')
    parser.add_argument('--filter_num', type=int, default=1,
            help='1: defulat filter size')
    parser.add_argument('--cnn_type', type=int, default=1,
            help='1: filter_size 7, 2: filter_size 3 and 7, 3: filter_size 3, 5 and 7')
    parser.add_argument('--agg', action='store_true',
            help='include agg')
    parser.add_argument('--sel', action='store_true',
            help='include sel')
    parser.add_argument('--cond', action='store_true',
            help='include cond')
    parser.add_argument('--detach', action='store_true',
            help='detach normal attention')
    args = parser.parse_args()

    do_test(args)