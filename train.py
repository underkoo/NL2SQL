import json
import torch
from programs.utils import *
from programs.model.predictor import Predictor
import numpy as np
import datetime

import argparse
import os

def do_train (args):
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
    TRAIN_ENTRY=(args.agg, args.sel, args.cond)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
    learning_rate = 1e-4

    dataset = 1
    # 0: original dataset, 1: re-split dataset

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=False, use_small=USE_SMALL)

    
    model = Predictor(word_emb, N_word=N_word, use_ca=args.ca, use_cnn=args.cnn, use_col_cnn=args.col_cnn,
            filter_num=args.filter_num, gpu=GPU, trainable_emb = False, agg=args.agg, sel=args.sel, cond=args.cond, use_detach=args.detach)
    optimizer = torch.optim.Adam(model.parameters(),
            lr=learning_rate, weight_decay = 0)

    for i in range(100):
        if args.toy:
            if not os.path.isfile("result/toy_train%d.txt"%i):
                result_file = open("result/toy_train%d.txt"%i, 'w')
                break;
        else:
            if not os.path.isfile("result/real_train%d.txt"%i):
                result_file = open("result/real_train%d.txt"%i, 'w')
                break;

    args_dic = vars(args)
    for arg in args_dic:
        result_file.write("%s = %s\n"%(arg, args_dic[arg]))

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(dataset, args)
    else:
        agg_m, sel_m, cond_m = best_model_name(dataset, args)

    if args.train_emb: # Load pretrained model.
        agg_lm, sel_lm, cond_lm = best_model_name(dataset, args, for_load=True)
        if args.agg:
            print ("Loading from %s"%agg_lm)
            result_file.write("Loading from %s"%agg_lm)
            model.agg_pred.load_state_dict(torch.load(agg_lm))
        if args.sel:
            print ("Loading from %s"%sel_lm)
            result_file.write("Loading from %s"%sel_lm)
            model.sel_pred.load_state_dict(torch.load(sel_lm))
        if args.cond:
            print ("Loading from %s"%cond_lm)
            result_file.write("Loading from %s"%cond_lm)
            model.cond_pred.load_state_dict(torch.load(cond_lm))

    init_acc = epoch_acc(model, BATCH_SIZE,
                val_sql_data, val_table_data, TRAIN_ENTRY)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print ('Init dev acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s'%\
            init_acc)
    result_file.write('Init dev acc_qm: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n'%\
            init_acc)
    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
        if args.train_emb:
            torch.save(model.agg_embed_layer.state_dict(), agg_e)
    if TRAIN_SEL:
        torch.save(model.sel_pred.state_dict(), sel_m)
        if args.train_emb:
            torch.save(model.sel_embed_layer.state_dict(), sel_e)
    if TRAIN_COND:
        torch.save(model.cond_pred.state_dict(), cond_m)
        if args.train_emb:
            torch.save(model.cond_embed_layer.state_dict(), cond_e)

    for i in range(args.epoch):
        print ('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
        result_file.write('Epoch %d @ %s\n'%(i+1, datetime.datetime.now()))
        loss = epoch_train(
                model, optimizer, BATCH_SIZE, 
                sql_data, table_data, TRAIN_ENTRY)
        print (' Loss = %s'%loss)
        result_file.write(' Loss = %s\n'%loss)
        # if i % 10 == 0:
        #     train_err_num = epoch_error(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY)
        #     print (' Train err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s'% train_err_num)
        #     result_file.write(' Train err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n'% train_err_num)
        #     val_err_num = epoch_error(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        #     print (' Dev err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s'% val_err_num)
        #     result_file.write(' Dev err num: %s;  breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n'% val_err_num)
        train_acc_qm = epoch_acc(model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY)
        print (' Train acc_qm: %s   breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s'% train_acc_qm)
        result_file.write(' Train acc_qm: %s   breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n'% train_acc_qm)
        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        print (' Dev acc_qm: %s   breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s'%val_acc)
        result_file.write(' Dev acc_qm: %s   breakdown on\n (agg, sel, where, cond_num, cond_col, cond_op, cond_val):\n %s\n'%val_acc)
        if args.agg and TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i+1
                torch.save(model.agg_pred.state_dict(),
                    'saved_model/epoch%d.agg_model%s'%(i+1, args.suffix))
                torch.save(model.agg_pred.state_dict(), agg_m)
                if args.train_emb:
                    torch.save(model.agg_embed_layer.state_dict(),
                    'saved_model/epoch%d.agg_embed%s'%(i+1, args.suffix))
                    torch.save(model.agg_embed_layer.state_dict(), agg_e)
        if args.sel and TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model.sel_pred.state_dict(),
                    'saved_model/epoch%d.sel_model%s'%(i+1, args.suffix))
                torch.save(model.sel_pred.state_dict(), sel_m)
                if args.train_emb:
                    torch.save(model.sel_embed_layer.state_dict(),
                    'saved_model/epoch%d.sel_embed%s'%(i+1, args.suffix))
                    torch.save(model.sel_embed_layer.state_dict(), sel_e)
        if args.cond and TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model.cond_pred.state_dict(),
                    'saved_model/epoch%d.cond_model%s'%(i+1, args.suffix))
                torch.save(model.cond_pred.state_dict(), cond_m)
                if args.train_emb:
                    torch.save(model.cond_embed_layer.state_dict(),
                    'saved_model/epoch%d.cond_embed%s'%(i+1, args.suffix))
                    torch.save(model.cond_embed_layer.state_dict(), cond_e)
        print (' Best val acc = %s, on epoch %s individually'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))
        result_file.write(' Best val acc = %s, on epoch %s individually\n'%(
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))

    result_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--suffix', type=str, default='',
            help='The suffix at the end of saved model name.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--train_emb', action='store_true',
            help='Train word embedding for SQLNet(requires pretrained model).')
    parser.add_argument('--cnn', action='store_true',
            help='Use cnn for predicting num of where clause')
    parser.add_argument('--col_cnn', action='store_true',
            help='apply cnn to column')
    parser.add_argument('--filter_num', type=int, default=1,
            help='1: defulat filter size')
    parser.add_argument('--agg', action='store_true',
            help='include agg')
    parser.add_argument('--sel', action='store_true',
            help='include sel')
    parser.add_argument('--cond', action='store_true',
            help='include cond')
    parser.add_argument('--detach', action='store_true',
            help='detach normal attention')
    parser.add_argument('--epoch', type=int, default=100,
            help='100 : default epoch')
    args = parser.parse_args()

    do_train(args)
