import numpy as np
import torch
import torch.nn as nn

from onir import vocab
from onir.vocab import WordvecVocab, WordvecHashVocab
from utils.aggregations import get_aggregation_func


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
