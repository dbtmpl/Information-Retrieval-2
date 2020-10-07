import torch
from torch import nn

from onir import util, rankers, predictors, datasets, modules
from onir.predictors import Reranker
from onir.rankers.conv_knrm import ConvKnrm
from onir.rankers.cedr_knrm import CedrKnrm

from onir.vocab import wordvec_vocab

from utils.general_utils import apply_spec_batch_qqa


@rankers.register('conv_knrm_qqa')
class ConvKnrmQQA(ConvKnrm):

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'mus': '-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0',
            'sigmas': '0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001',
            'grad_kernels': True,
            'max_ngram': 3,
            'crossmatch': True,
            'conv_filters': 128,
            'embed_dim': 300,
            'combine_channels': False,
            'pretrained_kernels': False,
        })
        return result

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({
            'query_tok', 'doc_tok', 'query_len', 'doc_len',
            'question_tok', 'answer_tok', 'question_len', 'answer_len',
        })
        return result


@predictors.register('reranker_qqa')
class RerankerQQA(Reranker):

    def _iter_batches(self, device):
        fields = set(self.input_spec['fields']) | {'query_id', 'doc_id'}
        it = datasets.record_iter(self.dataset,
                                  fields=fields,
                                  source=self.config['source'],
                                  run_threshold=self.config['run_threshold'],
                                  minrel=None,
                                  shuf=False,
                                  random=self.random,
                                  inf=False)
        for batch_items in util.chunked(it, self.config['batch_size']):
            batch = {}
            for record in batch_items:
                for k, seq in record.items():
                    batch.setdefault(k, []).append(seq)
            batch = apply_spec_batch_qqa(batch, self.input_spec, device)
            # ship 'em
            yield batch



@rankers.register('cedr_knrm_qqa')
class CedrKnrm(rankers.knrm.Knrm):
    """
    Implementation of CEDR for the KNRM model described in:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a model first trained using Vanilla BERT.
    TODO: additions made for QQA
    """

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)
        enc = self.encoder
        assert 'cls' in enc.enc_spec()['joint_fields'], \
               "CedrKnrm requires a vocabulary that supports CLS encoding, e.g., BertVocab"
        self.combine = nn.Linear(self.combine.in_features + enc.dim(), self.combine.out_features)

    def _forward(self, **inputs):
        rep = self.encoder.enc_query_doc(**inputs)
        simmat = self.simmat(rep['query'], rep['doc'], inputs['query_tok'], inputs['doc_tok'])
        kernel_scores = self.kernel_pool(simmat)
        all_scores = torch.cat([kernel_scores, rep['cls'][-1]], dim=1)
        return self.combine(all_scores)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({
            'query_tok', 'doc_tok', 'query_len', 'doc_len',
            'question_tok', 'answer_tok', 'question_len', 'answer_len',
        })
        return result