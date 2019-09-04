import os
import sys
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from sru import *
import dataloader
import modules

class Model(nn.Module):
    def __init__(self, embedding, hidden_size=150, depth=1, dropout=0.3, cnn=False, nclasses=2):
        super(Model, self).__init__()
        self.cnn = cnn
        self.drop = nn.Dropout(dropout)
        self.emb_layer = modules.EmbeddingLayer(
            embs = dataloader.load_embedding(embedding)
        )
        self.word2id = self.emb_layer.word2id

        if cnn:
            self.encoder = modules.CNN_Text(
                self.emb_layer.n_d,
                widths = [3,4,5],
                filters=hidden_size
            )
            d_out = 3*hidden_size
        else:
            self.encoder = nn.LSTM(
                self.emb_layer.n_d,
                hidden_size//2,
                depth,
                dropout = dropout,
                # batch_first=True,
                bidirectional=True
            )
            d_out = hidden_size
        # else:
        #     self.encoder = SRU(
        #         emb_layer.n_d,
        #         args.d,
        #         args.depth,
        #         dropout = args.dropout,
        #     )
        #     d_out = args.d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)

        if self.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            # output = output[-1]
            output = torch.max(output, dim=0)[0].squeeze()

        output = self.drop(output)
        return self.out(output)

    def text_pred(self, text, batch_size=32):
        batches_x = dataloader.create_batches_x(
            text,
            batch_size, ##TODO
            self.word2id
        )
        outs = []
        with torch.no_grad():
            for x in batches_x:
                x = Variable(x)
                if self.cnn:
                    x = x.t()
                emb = self.emb_layer(x)

                if self.cnn:
                    output = self.encoder(emb)
                else:
                    output, hidden = self.encoder(emb)
                    # output = output[-1]
                    output = torch.max(output, dim=0)[0]

                outs.append(F.softmax(self.out(output), dim=-1))

        return torch.cat(outs, dim=0)


def eval_model(niter, model, input_x, input_y):
    model.eval()
    # N = len(valid_x)
    # criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.
    # total_loss = 0.0
    with torch.no_grad():
        for x, y in zip(input_x, input_y):
            x, y = Variable(x, volatile=True), Variable(y)
            output = model(x)
            # loss = criterion(output, y)
            # total_loss += loss.item()*x.size(1)
            pred = output.data.max(1)[1]
            correct += pred.eq(y.data).cpu().sum()
            cnt += y.numel()
    model.train()
    return correct.item()/cnt

def train_model(epoch, model, optimizer,
        train_x, train_y,
        test_x, test_y,
        best_test, save_path):

    model.train()
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    test_acc = eval_model(niter, model, test_x, test_y)

    sys.stdout.write("Epoch={} iter={} lr={:.6f} train_loss={:.6f} test_err={:.6f}\n".format(
        epoch, niter,
        optimizer.param_groups[0]['lr'],
        loss.item(),
        test_acc
    ))

    if test_acc > best_test:
        best_test = test_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
        # test_err = eval_model(niter, model, test_x, test_y)
    sys.stdout.write("\n")
    return best_test

def save_data(data, labels, path, type='train'):
    with open(os.path.join(path, type+'.txt'), 'w') as ofile:
        for text, label in zip(data, labels):
            ofile.write('{} {}\n'.format(label, ' '.join(text)))

def main(args):
    if args.dataset == 'mr':
    #     data, label = dataloader.read_MR(args.path)
    #     train_x, train_y, test_x, test_y = dataloader.cv_split2(
    #         data, label,
    #         nfold=10,
    #         valid_id=args.cv
    #     )
    #
    #     if args.save_data_split:
    #         save_data(train_x, train_y, args.path, 'train')
    #         save_data(test_x, test_y, args.path, 'test')
        train_x, train_y = dataloader.read_corpus('/data/medg/misc/jindi/nlp/datasets/mr/train.txt')
        test_x, test_y = dataloader.read_corpus('/data/medg/misc/jindi/nlp/datasets/mr/test.txt')
    elif args.dataset == 'imdb':
        train_x, train_y = dataloader.read_corpus(os.path.join('/data/medg/misc/jindi/nlp/datasets/imdb',
                                                               'train_tok.csv'),
                                                  clean=False, MR=True, shuffle=True)
        test_x, test_y = dataloader.read_corpus(os.path.join('/data/medg/misc/jindi/nlp/datasets/imdb',
                                                               'test_tok.csv'),
                                                clean=False, MR=True, shuffle=True)
    else:
        train_x, train_y = dataloader.read_corpus('/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/'
                                                    'train_tok.csv'.format(args.dataset),
                                                  clean=False, MR=False, shuffle=True)
        test_x, test_y = dataloader.read_corpus('/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/'
                                                    'test_tok.csv'.format(args.dataset),
                                                clean=False, MR=False, shuffle=True)

    nclasses = max(train_y) + 1
    # elif args.dataset == 'subj':
    #     data, label = dataloader.read_SUBJ(args.path)
    # elif args.dataset == 'cr':
    #     data, label = dataloader.read_CR(args.path)
    # elif args.dataset == 'mpqa':
    #     data, label = dataloader.read_MPQA(args.path)
    # elif args.dataset == 'trec':
    #     train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path)
    #     data = train_x + test_x
    #     label = None
    # elif args.dataset == 'sst':
    #     train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path)
    #     data = train_x + valid_x + test_x
    #     label = None
    # else:
    #     raise Exception("unknown dataset: {}".format(args.dataset))

    # if args.dataset == 'trec':


    # elif args.dataset != 'sst':
    #     train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.cv_split(
    #         data, label,
    #         nfold = 10,
    #         test_id = args.cv
    #     )

    model = Model(args.embedding, args.d, args.depth, args.dropout, args.cnn, nclasses).cuda()
    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    train_x, train_y = dataloader.create_batches(
        train_x, train_y,
        args.batch_size,
        model.word2id,
    )
    # valid_x, valid_y = dataloader.create_batches(
    #     valid_x, valid_y,
    #     args.batch_size,
    #     emb_layer.word2id,
    # )
    test_x, test_y = dataloader.create_batches(
        test_x, test_y,
        args.batch_size,
        model.word2id,
    )

    best_test = 0
    # test_err = 1e+8
    for epoch in range(args.max_epoch):
        best_test = train_model(epoch, model, optimizer,
            train_x, train_y,
            # valid_x, valid_y,
            test_x, test_y,
            best_test, args.save_path
        )
        if args.lr_decay>0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

    # sys.stdout.write("best_valid: {:.6f}\n".format(
    #     best_valid
    # ))
    sys.stdout.write("test_err: {:.6f}\n".format(
        best_test
    ))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print (args)
    torch.cuda.set_device(args.gpu_id)
    main(args)