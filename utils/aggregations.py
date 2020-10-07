import torch


def get_aggregation_func(key):
    aggregations = {
        'mean': mean_aggregation,
        'concat': concat_aggregation,
        'weighted': weighted_aggregation,
    }
    return aggregations[key]


def mean_aggregation(query_embed, question_embed, answer_embed, dim=2):
    stacked_embed = torch.stack((query_embed, question_embed, answer_embed), dim=dim)
    return torch.mean(stacked_embed, dim=dim)


def concat_aggregation(query_embed, question_embed, answer_embed, dim=2):
    return torch.cat((query_embed, question_embed, answer_embed), dim=dim)


def weighted_aggregation(query_embed, question_embed, answer_embed):
    return (2 / 3) * query_embed + (1 / 6) * question_embed + (1 / 6) * answer_embed
