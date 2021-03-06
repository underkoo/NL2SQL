from train import do_train
from test import do_test
import argparse

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
    parser.add_argument('--num_cnn', action='store_true',
            help='apply cnn to the number of columns')
    parser.add_argument('--op_cnn', action='store_true',
            help='apply cnn to operator')
    parser.add_argument('--val_cnn', action='store_true',
            help='apply cnn to value')
    parser.add_argument('--agg_cnn', action='store_true',
            help='apply cnn to aggregator')
    parser.add_argument('--sel_cnn', action='store_true',
            help='apply cnn to sel')
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
    parser.add_argument('--epoch', type=int, default=100,
            help='100 : default epoch')
    args = parser.parse_args()

    do_train(args)
    do_test(args)