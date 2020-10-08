import torch

from onir import rankers
from onir.rankers.vanilla_transformer import VanillaTransformer

from utils.aggregations import get_aggregation_func


@rankers.register('vanilla_transformer_qqa_joint')
class VanillaTransformerQQAJoint(VanillaTransformer):

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({
            'query_tok', 'query_len', 'doc_tok', 'doc_len',
            'question_tok', 'answer_tok', 'question_len', 'answer_len'
        })
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result


@rankers.register('vanilla_transformer_qqa_sep')
class VanillaTransformerQQASep(VanillaTransformer):

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({
            'query_tok', 'query_len', 'doc_tok', 'doc_len',
            'question_tok', 'answer_tok', 'question_len', 'answer_len'
        })
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result

    def _forward(self, **inputs):
        pooled_output = self.encoder.enc_query_doc(**inputs)
        qqa_cls = pooled_output['qqa_cls']
        doc_cls = pooled_output['doc_cls'][-1]
        pooled_output = self._doc_qqa_aggregation(qqa_cls, doc_cls)

        pooled_output = self.dropout(pooled_output)
        result = self.ranker(pooled_output)
        if self.config['combine'] == 'prob':
            result = result.softmax(dim=1)[:, 1]
        elif self.config['combine'] == 'logprob':
            result = result.log_softmax(dim=1)[:, 1]
        return result

    def _doc_qqa_aggregation(self, qqa_cls, doc_cls, dim=1):
        # TODO: Think about clever aggregations. Note we already aggregated QQA.
        stacked_embed = torch.stack((qqa_cls, doc_cls), dim=dim)
        return torch.mean(stacked_embed, dim=dim)
