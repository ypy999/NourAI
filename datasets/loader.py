from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.configuration import ProjectConfig, load_config
from tokenizer.byte_tokenizer import ByteTokenizer
from .dataset import TokenSequenceDataset


def create_dataloaders(processed_dir, config=None, tokenizer=None, num_workers=0):
    tokenizer = tokenizer or ByteTokenizer()
    project = config if isinstance(config, ProjectConfig) else load_config(config, tokenizer)
    project.validate(tokenizer)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    root = Path(processed_dir)
    train = TokenSequenceDataset(root / "train.jsonl", project, tokenizer)
    validation = TokenSequenceDataset(root / "validation.jsonl", project, tokenizer)
    train_generator = torch.Generator().manual_seed(project.random_seed)
    validation_generator = torch.Generator().manual_seed(project.random_seed + 1)
    return (
        DataLoader(train, batch_size=project.batch_size, shuffle=True, num_workers=num_workers, generator=train_generator),
        DataLoader(validation, batch_size=project.batch_size, shuffle=False, num_workers=num_workers, generator=validation_generator),
    )
