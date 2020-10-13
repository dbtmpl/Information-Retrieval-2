import torch
import torch.nn as nn
from torch.nn import functional as F

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

        conv_channels = [8, 16]
        self.doc_embed = []
        cur_channels = 1
        for k, conv_dim in enumerate(conv_channels):
            conv = nn.Conv1d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.add_module("doc_embed{}".format(k + 1), conv)
            self.doc_embed.append(conv)
            cur_channels = conv_dim

        self.qqa_doc_embed = []
        conv_channels = [16, 4]
        # 2x because we concatenate qqa and doc embedding
        cur_channels = conv_channels[0] * 2
        for k, conv_dim in enumerate(conv_channels):
            conv = nn.Conv1d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.add_module("qqa_doc_embed{}".format(k + 1), conv)
            self.qqa_doc_embed.append(conv)
            cur_channels = conv_dim

    def _forward(self, **inputs):
        pooled_output = self.encoder.enc_query_doc(**inputs)
        qqa_embed = pooled_output['qqa_cls']
        doc_cls = pooled_output['doc_cls'][-1][:, None, :]

        qqa_doc_embed = self._doc_qqa_aggregation(qqa_embed, doc_cls)
        qqa_doc_embed = self.dropout(qqa_doc_embed)

        bs, c, l = qqa_doc_embed.size()
        for layer in self.qqa_doc_embed:
            qqa_doc_embed = layer(qqa_doc_embed)

        qqa_doc_embed = qqa_doc_embed.view(bs, -1).squeeze()

        result = self.ranker(qqa_doc_embed)

        if self.config['combine'] == 'prob':
            result = result.softmax(dim=1)[:, 1]
        elif self.config['combine'] == 'logprob':
            result = result.log_softmax(dim=1)[:, 1]
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

    def _doc_qqa_aggregation(self, qqa_embed, doc_embed):
        for layer in self.doc_embed:
            doc_embed = F.relu(layer(doc_embed))
        return torch.cat((qqa_embed, doc_embed), dim=1)
