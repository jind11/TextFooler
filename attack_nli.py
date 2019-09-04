import sys
import argparse
import os
import numpy as np
import fnmatch
import criteria
import string
import pickle
import random

from InferSent.models import NLINet
from esim.model import ESIM
from esim.data import Preprocessor
from esim.utils import correct_predictions

import tensorflow as tf
import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig


class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()

        #         self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cpu")
        # torch.cuda.set_device(local_rank)

        # Retrieving model parameters from checkpoint.
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 1,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }

        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(pretrained_file, map_location='cuda:0'))

        # construct dataset loader
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        checkpoint = torch.load(pretrained_file)
        # Retrieving model parameters from checkpoint.
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])

        # construct dataset loader
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


# class NLIDataset_InferSent(Dataset):
#     """
#     Dataset class for Natural Language Inference datasets.
#
#     The class can be used to read preprocessed datasets where the premises,
#     hypotheses and labels have been transformed to unique integer indices
#     (this can be done with the 'preprocess_data' script in the 'scripts'
#     folder of this repository).
#     """
#
#     def __init__(self,
#                  embedding_path,
#                  dataset='SNLI',
#                  word_emb_dim=300,
#                  batch_size=32,
#                  bos="<s>",
#                  eos="</s>"):
#         """
#         Args:
#             data: A dictionary containing the preprocessed premises,
#                 hypotheses and labels of some dataset.
#             padding_idx: An integer indicating the index being used for the
#                 padding token in the preprocessed data. Defaults to 0.
#             max_premise_length: An integer indicating the maximum length
#                 accepted for the sequences in the premises. If set to None,
#                 the length of the longest premise in 'data' is used.
#                 Defaults to None.
#             max_hypothesis_length: An integer indicating the maximum length
#                 accepted for the sequences in the hypotheses. If set to None,
#                 the length of the longest hypothesis in 'data' is used.
#                 Defaults to None.
#         """
#         self.bos = bos
#         self.eos = eos
#         self.word_emb_dim = word_emb_dim
#         self.batch_size = batch_size
#
#         # read all data
#         files = []
#         if dataset == 'SNLI':
#             data_dir = '/data/medg/misc/jindi/nlp/datasets/SNLI/snli_1.0'
#             for file in os.listdir(data_dir):
#                 if fnmatch.fnmatch(file, '*_train.txt') or \
#                         fnmatch.fnmatch(file, '*_dev.txt') or \
#                         fnmatch.fnmatch(file, '*_test.txt'):
#                     files.append(file)
#         else:
#             data_dir = '/data/medg/misc/jindi/nlp/datasets/MNLI'
#             for file in os.listdir(data_dir):
#                 if fnmatch.fnmatch(file, '*_train.txt') or \
#                         fnmatch.fnmatch(file, '*_dev_matched.txt') or \
#                         fnmatch.fnmatch(file, '*_dev_mismatched.txt'):
#                     files.append(file)
#
#         data = []
#         for file in files:
#             data_tmp = read_data(os.path.join(data_dir, file))
#             data.extend(data_tmp['premises'] + data_tmp['hypotheses'])
#
#         # build word dict
#         self.word_vec = self.build_vocab(data, embedding_path)
#
#     def build_vocab(self, sentences, embedding_path):
#         word_dict = self.get_word_dict(sentences)
#         word_vec = self.get_embedding(word_dict, embedding_path)
#         print('Vocab size : {0}'.format(len(word_vec)))
#         return word_vec
#
#     def get_word_dict(self, sentences):
#         # create vocab of words
#         word_dict = {}
#         for sent in sentences:
#             for word in sent:
#                 if word not in word_dict:
#                     word_dict[word] = ''
#         word_dict['<s>'] = ''
#         word_dict['</s>'] = ''
#         word_dict['<oov>'] = ''
#         return word_dict
#
#     def get_embedding(self, word_dict, embedding_path):
#         # create word_vec with glove vectors
#         word_vec = {}
#         word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
#         with open(embedding_path) as f:
#             for line in f:
#                 word, vec = line.split(' ', 1)
#                 if word in word_dict:
#                     word_vec[word] = np.array(list(map(float, vec.split())))
#         print('Found {0}(/{1}) words with embedding vectors'.format(
#             len(word_vec), len(word_dict)))
#         return word_vec
#
#     def get_batch(self, batch, word_vec, emb_dim=300):
#         # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
#         lengths = np.array([len(x) for x in batch])
#         max_len = np.max(lengths)
#         #         print(max_len)
#         embed = np.zeros((max_len, len(batch), emb_dim))
#
#         for i in range(len(batch)):
#             for j in range(len(batch[i])):
#                 if batch[i][j] in word_vec:
#                     embed[j, i, :] = word_vec[batch[i][j]]
#                 else:
#                     embed[j, i, :] = word_vec['<oov>']
#         #                     embed[j, i, :] = np.random.normal(size=(emb_dim))
#
#         return torch.from_numpy(embed).float(), lengths
#
#     def transform_text(self, data):
#         # transform data into seq of embeddings
#         premises = data['premises']
#         hypotheses = data['hypotheses']
#
#         # add bos and eos
#         premises = [['<s>'] + premise + ['</s>'] for premise in premises]
#         hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]
#
#         batches = []
#         for stidx in range(0, len(premises), self.batch_size):
#             # prepare batch
#             s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
#                                               self.word_vec, self.word_emb_dim)
#             s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
#                                               self.word_vec, self.word_emb_dim)
#             batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))
#
#         return batches


class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader


def attack(premise, hypothese, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(hypothese)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # get importance score
        leave_1_texts = [hypothese[:ii]+['<oov>']+hypothese[min(ii+1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor({'premises':[premise]*len_text, 'hypotheses': leave_1_texts})
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance scorefor word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            if score > import_score_threshold and hypothese[idx] not in stop_words_set:
                words_perturb.append((idx, hypothese[idx]))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts})

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        return ' '.join(text_prime), num_changed, orig_label, \
               torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]})), num_queries


def random_attack(premise, hypothese, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
           sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
           synonym_num=50, batch_size=32):
    # first check the prediction of the original text
    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypothese]}).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(hypothese)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(hypothese)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, hypothese[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = hypothese[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = predictor({'premises': [premise] * len(synonyms), 'hypotheses': new_texts})

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                    (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]

        return ' '.join(text_prime), num_changed, orig_label, \
               torch.argmax(predictor({'premises':[premise], 'hypotheses': [text_prime]})), num_queries


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")


    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack, fetch first [args.data_size] data samples for adversarial attacking
    data = read_data(args.dataset_path, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                                args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}

    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')
    orig_premises = []
    orig_hypotheses = []
    adv_hypotheses = []
    true_labels = []
    new_labels = []

    stop_words_set = criteria.get_stopwords()
    for idx, premise in enumerate(data['premises']):
        if idx % 100 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(premise, hypothese, true_label, predictor,
                                                    args.perturb_ratio, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)
        else:
            new_text, num_changed, orig_label, \
            new_label, num_queries = attack(premise, hypothese, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, cos_sim, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)
        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(hypothese)
        # print('orig sentence ({}):'.format(orig_label), ' '.join(text), '\nto new sentence ({}):'.format(new_label),
        #       new_text, '\n{}/{} changed at {:.2f}%'.format(num_changed, len(text), changed_rate * 100))
        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_premises.append(' '.join(premise))
            orig_hypotheses.append(' '.join(hypothese))
            adv_hypotheses.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label.item())

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-adv_failures/1000)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)
    log_file.write(message)


    if args.target_model == 'bert':
        labeldict = {0: "contradiction",
                     1: "entailment",
                     2:  "neutral"}
    else:
        labeldict = {0: "entailment",
                     1: "neutral",
                     2: "contradiction"}

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_premise, orig_hypothesis, adv_hypothesis, \
            true_label, new_label in zip(orig_premises, orig_hypotheses, adv_hypotheses,
                                        true_labels, new_labels):
            ofile.write('orig premise:\t{}\norig hypothesis ({}):\t{}\n'
                        'adv hypothesis ({}):\t{}\n\n'.format(orig_premise,
                                                              labeldict[true_label],
                                                              orig_hypothesis,
                                                              labeldict[new_label],
                                                              adv_hypothesis))

if __name__ == "__main__":
    main()