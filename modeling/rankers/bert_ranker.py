from onir import rankers
from onir.rankers.vanilla_transformer import VanillaTransformer

from utils.aggregations import get_aggregation_func


@rankers.register('vanilla_transformer_qqa')
class VanillaTransformerQQA(VanillaTransformer):

    def __init__(self, vocab, config, logger, random):
        super().__init__(vocab, config, logger, random)
        self.enc_aggregation = config['aggregation']
        self.aggregation_func = get_aggregation_func(self.enc_aggregation)

    @staticmethod
    def default_config():
        result = rankers.Ranker.default_config()
        result.update({
            'combine': 'linear',  # one of linear, prob
            'outputs': 1,
            'aggregation': 'mean'
        })
        return result

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
        pooled_output = self._aggregate_outputs(pooled_output)
        pooled_output = self.dropout(pooled_output)
        result = self.ranker(pooled_output)
        if self.config['combine'] == 'prob':
            result = result.softmax(dim=1)[:, 1]
        elif self.config['combine'] == 'logprob':
            result = result.log_softmax(dim=1)[:, 1]
        return result

    def _aggregate_outputs(self, outputs):
        query_cls = outputs['query_cls'][-1]
        question_cls = outputs['question_cls'][-1]
        answer_cls = outputs['answer_cls'][-1]
        aggregation = self.aggregation_func(query_cls, question_cls, answer_cls, dim=1)
        return aggregation
