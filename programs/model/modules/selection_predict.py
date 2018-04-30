import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from programs.model.modules.net_utils import run_lstm, col_name_encode

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca, use_cnn, filter_num):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.use_cnn = use_cnn
        self.filter_num = filter_num
        self.max_tok_num = max_tok_num
        if use_cnn:
            self.sel_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.filter_num,
                    kernel_size= (7, N_word),
                    stride= (1, 1),
                    padding= (3, 0)
                ),
                nn.BatchNorm2d(self.filter_num),
                nn.RReLU()
            )
            self.sel_dropout = nn.Dropout2d(p=0.5)
            self.sel_name_enc = nn.LSTM(input_size=N_word,
                    hidden_size=int(self.filter_num/2), num_layers=N_depth,
                    batch_first=True, dropout=0.5, bidirectional=True)
            self.sel_att = nn.Linear(self.filter_num, 1)    


            self.sel_out_K = nn.Linear(self.filter_num, self.filter_num)
            self.sel_out_col = nn.Linear(self.filter_num, self.filter_num)
            self.sel_out  = nn.Linear(self.filter_num, 1)
        else:
            self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                    num_layers=N_depth, batch_first=True,
                    dropout=0.3, bidirectional=True)
            if use_ca:
                print ("Using column attention on selection predicting")
                self.sel_att = nn.Linear(N_h, N_h)
            else:
                print ("Not using column attention on selection predicting")
                self.sel_att = nn.Linear(N_h, 1)
            self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                    num_layers=N_depth, batch_first=True,
                    dropout=0.3, bidirectional=True)
            self.sel_out_K = nn.Linear(N_h, N_h)
            self.sel_out_col = nn.Linear(N_h, N_h)
            self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
            
        self.softmax = nn.Softmax()


    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        if self.use_cnn:
            e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.sel_name_enc)
            x_cond_col = torch.unsqueeze(x_emb_var, dim=1)
            sel_conv_h = self.sel_conv(x_cond_col)
            sel_h = sel_conv_h.squeeze()
            sel_att_val = torch.bmm(e_cond_col, sel_h)
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    sel_att_val[idx, :, num:] = -100
            sel_att = self.softmax(sel_att_val.view(
                (-1, np.asscalar(max_x_len)))).view(B, -1, np.asscalar(max_x_len))
            sel_val = (sel_h.transpose(1,2).unsqueeze(1) * sel_att.unsqueeze(3)).sum(2)
            sel_score = self.sel_out(sel_val).squeeze()
            max_col_num = max(col_num)
            for b, num in enumerate(col_num):
                if num < max_col_num:
                    sel_score[b, num:] = -100
        else:
            e_col, _ = col_name_encode(col_inp_var, col_name_len,
                    col_len, self.sel_col_name_enc)

            if self.use_ca:
                h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
                att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
                for idx, num in enumerate(x_len):
                    if num < max_x_len:
                        att_val[idx, :, num:] = -100
                att = self.softmax(att_val.view((-1, np.asscalar(max_x_len)))).view(
                        B, -1, np.asscalar(max_x_len))
                K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
            else:
                h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
                att_val = self.sel_att(h_enc).squeeze()
                for idx, num in enumerate(x_len):
                    if num < max_x_len:
                        att_val[idx, num:] = -100
                att = self.softmax(att_val)
                K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
                K_sel_expand=K_sel.unsqueeze(1)

            sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                    self.sel_out_col(e_col) ).squeeze()
            max_col_num = max(col_num)
            for idx, num in enumerate(col_num):
                if num < max_col_num:
                    sel_score[idx, num:] = -100
        return sel_score
