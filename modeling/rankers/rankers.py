from onir import util, rankers, predictors, datasets
from onir.predictors import Reranker
from onir.rankers.conv_knrm import ConvKnrm
from onir.rankers.pacrr import Pacrr

from utils.general_utils import apply_spec_batch_qqa


@rankers.register('pacrr_qqa')
class PaccrQQA(Pacrr):

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({
            'query_tok', 'doc_tok', 'query_len', 'doc_len',
            'question_tok', 'answer_tok', 'question_len', 'answer_len',
        })
        return result


@rankers.register('conv_knrm_qqa')
class ConvKnrmQQA(ConvKnrm):

    @staticmethod
    def default_config():
        result = ConvKnrm.default_config().copy()
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
