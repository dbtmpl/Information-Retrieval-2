import re
import hashlib

import numpy as np
import torch
import torch.nn as nn

from onir import vocab
from onir.vocab import WordvecVocab, WordvecHashVocab


@vocab.register('wordvec_hash_qqa')
class WordvecHashVocabQQA(WordvecHashVocab):

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
            stacked_embed = torch.stack((query_embed, question_embed, answer_embed), dim=2)

            return torch.mean(stacked_embed, dim=2)


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
