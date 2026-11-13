import torch


def create_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        betas=tuple(config.adam_betas), eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
