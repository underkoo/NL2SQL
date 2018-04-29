# NL2SQL

[WikiSQL dataset](https://github.com/salesforce/WikiSQL)

This repository contains implemenations of SQLNet and CNN-based NL2SQL model.

## Citation

> Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning.

## Bibtex

```
@article{xu2017sqlnet,
  title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning},
  author={Xu, Xiaojun and Liu, Chang and Song, Dawn},
  journal={arXiv preprint arXiv:1711.04436},
  year={2017}
}
```

## Installation
The data is in `data.tar.bz2`. Unzip the code by running
```bash
tar -xjvf data.tar.bz2
```

The code is written using PyTorch in Python 2.7. Check [here](http://pytorch.org/) to install PyTorch. You can install other dependency by running 
```bash
pip install -r requirements.txt
```

## Downloading the glove embedding.
Download the pretrained glove embedding from [here](https://github.com/stanfordnlp/GloVe) using
```bash
bash download_glove.sh
```

## Train
The training script is `train.py`. To see the detailed parameters for running:
```bash
python train.py -h
```

Some typical usage are listed as below:

Train a SQLNet model
```bash
python train.py
```

Train a SQLNet model with column attention:
```bash
python train.py --ca
```

Train a CNN-based model:
```bash
python train.py --cnn --filter_num 256
```

## Test

Test a trained SQLNet model
```bash
python test.py
```

Test a trained SQLNet model with column attention
```bash
python test.py --ca
```

Test a trained and CNN-based model:
```bash
python test.py --cnn --filter_num 256
```

## Train and Test

You can do train and test the model sequentially
```bash
python tandt.py --cnn --filter_num 256
```