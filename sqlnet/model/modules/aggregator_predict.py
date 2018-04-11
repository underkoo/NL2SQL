import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sqlnet.model.modules.net_utils import run_lstm, col_name_encode



class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca, use_cnn, filter_size):
        super(AggPredictor, self).__init__()
        self.use_ca = use_ca
        self.use_cnn = use_cnn
        self.filter_size = filter_size
        if self.use_cnn:
            self.agg_conv1 = nn.Sequential(         # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.filter_size*4,
                    kernel_size= (3, 1),
                    stride= (1, 1),
                    padding= (1, 0)                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, 28, 28)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(kernel_size= (2, 1)),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
            self.agg_conv2 = nn.Sequential(         # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=self.filter_size*4,
                    out_channels=self.filter_size*8,
                    kernel_size= (3, 1),
                    stride= (1, 1),
                    padding= (1, 0)                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, 28, 28)
                nn.ReLU(),                      # activation
                nn.MaxPool2d(kernel_size= (2, 1)),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
            self.agg_conv_last = nn.Sequential(         # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=self.filter_size*8,
                    out_channels=self.filter_size*16,
                    kernel_size= (3, 1),
                    stride= (1, 1),
                    padding= (1, 0)                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, 28, 28)
                nn.ReLU(),                      # activation
                nn.AdaptiveMaxPool2d((8, N_word)),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
            self.agg_fc = nn.Sequential( # fully connected layer  
                nn.Linear(self.filter_size*16 * 8 * N_word, int(self.filter_size*16 * 8 * N_word / 2)),
                nn.ReLU()                      # activation
            )
            self.agg_fc_out = nn.Linear(int(self.filter_size*16 * 8 * N_word / 2), 6)

        else:
            self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                    num_layers=N_depth, batch_first=True,
                    dropout=0.3, bidirectional=True)
            if use_ca:
                print ("Using column attention on aggregator predicting")
                self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                        hidden_size=int(N_h/2), num_layers=N_depth,
                        batch_first=True, dropout=0.3, bidirectional=True)
                self.agg_att = nn.Linear(N_h, N_h)
            else:
                print ("Not using column attention on aggregator predicting")
                self.agg_att = nn.Linear(N_h, 1)
            self.agg_out = nn.Sequential(nn.Linear(N_h, N_h),
                    nn.Tanh(), nn.Linear(N_h, 6))
            self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
            col_len=None, col_num=None, gt_sel=None):
        if self.use_cnn:
            x_emb_var_dim = torch.unsqueeze(x_emb_var, dim=1)
            agg_conv1 = self.agg_conv1(x_emb_var_dim)
            agg_conv2 = self.agg_conv2(agg_conv1)
            agg_conv_last = self.agg_conv_last(agg_conv2)
            agg_h_conv = agg_conv_last.view(agg_conv_last.size(0), -1)
            agg_h_conv_fc = self.agg_fc(agg_h_conv)
            agg_score = self.agg_fc_out(agg_h_conv_fc)
        else:
            B = len(x_emb_var)
            max_x_len = max(x_len)

            h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
            if self.use_ca:
                e_col, _ = col_name_encode(col_inp_var, col_name_len, 
                        col_len, self.agg_col_name_enc)
                chosen_sel_idx = torch.LongTensor(gt_sel)
                aux_range = torch.LongTensor(range(len(gt_sel)))
                if x_emb_var.is_cuda:
                    chosen_sel_idx = chosen_sel_idx.cuda()
                    aux_range = aux_range.cuda()
                chosen_e_col = e_col[aux_range, chosen_sel_idx]
                att_val = torch.bmm(self.agg_att(h_enc), 
                        chosen_e_col.unsqueeze(2)).squeeze()
            else:
                att_val = self.agg_att(h_enc).squeeze()

            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)

            K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            agg_score = self.agg_out(K_agg)
        return agg_score
