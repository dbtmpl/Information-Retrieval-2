import re
import hashlib
import tempfile
import os
import tokenizers as tk

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer

from onir import vocab, config
from onir.interfaces import bert_models
from onir.vocab import WordvecVocab, WordvecHashVocab, BertVocab


@vocab.register('wordvec_hash_qqa')
class WordvecHashVocabQQA(WordvecHashVocab):

    def __init__(self, config, logger, random):
        super().__init__(config, logger, random)
        self.enc_aggregation = config['aggregation']

    @staticmethod
    def default_config():
        result = WordvecVocab.default_config().copy()
        result.update({
            'hashspace': 1000,
            'init_stddev': 0.5,
            'log_miss': False,
            'aggregation': 'mean'
        })
        return result

    def encoder(self):
        return WordvecEncoderQQA(self)


class WordvecEncoderQQA(vocab.VocabEncoder):

    def __init__(self, vocabulary):
        super().__init__(vocabulary)
        matrix = vocabulary._weights
        self.size = matrix.shape[1]
        matrix = np.concatenate([np.zeros((1, self.size)), matrix])  # add padding record (-1)
        self.embed = nn.Embedding.from_pretrained(
            torch.from_numpy(matrix.astype(np.float32)),
            freeze=not vocabulary.config['train']
        )
        self.aggregation_func = get_aggregation_func(vocabulary.enc_aggregation)

    def forward(self, toks, lens=None):

        if len(toks) == 1:
            # In this case we have documents
            return self.embed(toks[0] + 1)  # +1 to handle padding at position -1

        else:
            # In this case we have QQA, we do some naive aggregation
            query_toks, question_toks, answer_toks = toks

            query_embed = self.embed(query_toks + 1)
            question_embed = self.embed(question_toks + 1)
            answer_embed = self.embed(answer_toks + 1)

            return self.aggregation_func(query_embed, question_embed, answer_embed)

    def _enc_spec(self) -> dict:
        return {
            'dim': self.size,
            'views': 1,
            'static': True,
            'vocab_size': self.embed.weight.shape[0]
        }

    def enc_query_doc(self, **inputs):
        """
        Returns encoded versions of the query and document from general **inputs dict
        Requires query_tok, doc_tok, query_len, and doc_len.
        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """

        return {
            'query': self((inputs['query_tok'], inputs['question_tok'], inputs['answer_tok']), inputs['query_len']),
            'doc': self((inputs['doc_tok'],), inputs['doc_len'])
        }


def mean_aggregation(query_embed, question_embed, answer_embed):
    stacked_embed = torch.stack((query_embed, question_embed, answer_embed), dim=2)
    return torch.mean(stacked_embed, dim=2)


def concat_aggregation(query_embed, question_embed, answer_embed):
    return torch.cat((query_embed, question_embed, answer_embed), dim=2)


def weighted_aggregation(query_embed, question_embed, answer_embed):
    return (2 / 3) * query_embed + (1 / 6) * question_embed + (1 / 6) * answer_embed


def get_aggregation_func(key):
    aggregations = {
        'mean': mean_aggregation,
        'concat': concat_aggregation,
        'weighted': weighted_aggregation,
    }
    return aggregations[key]




#########


@vocab.register('bert_qqa')
class BertQQAVocab(BertVocab):
    @staticmethod
    def default_config():
        return {
            'bert_base': 'bert-base-uncased',
            'bert_weights': '',     # TODO: merge bert_base and bert_weights somehow, better integrate fine-tuning BERT into pipeline
            'layer': -1, # all layers
            'last_layer': False,
            'train': False,
            'encoding': config.Choices(['joint', 'sep']),
        }

    def __init__(self, config, logger):
        super().__init__(config, logger)
        bert_model = bert_models.get_model(config['bert_base'], self.logger)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        # HACK! Until the transformers library adopts tokenizers, save and re-load vocab
        with tempfile.TemporaryDirectory() as d:
            self.tokenizer.save_vocabulary(d)
            # this tokenizer is ~4x faster as the BertTokenizer, per my measurements
            self.tokenizer = tk.BertWordPieceTokenizer(os.path.join(d, 'vocab.txt'))

    def tokenize(self, text):
        # return self.tokenizer.tokenize(text)
        return self.tokenizer.encode(text).tokens[1:-1] # removes leading [CLS] and trailing [SEP]

    def tok2id(self, tok):
        # return self.tokenizer.vocab[tok]
        return self.tokenizer.token_to_id(tok)

    def id2tok(self, idx):
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        # return self.tokenizer.ids_to_tokens[idx]
        return self.tokenizer.id_to_token(idx)

    def lexicon_path_segment(self):
        return 'bert_{bert_base}'.format(**self.config)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()