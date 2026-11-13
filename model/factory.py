"""Single entry point for constructing a model from project JSON settings."""
from pathlib import Path

from configs.configuration import ProjectConfig, load_config
from model.transformer import GPT, GPTConfig
from tokenizer.byte_tokenizer import ByteTokenizer


def create_model(config_path=None, tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    project = config_path if isinstance(config_path, ProjectConfig) else load_config(config_path, tokenizer)
    project.validate(tokenizer)
    model_config = GPTConfig(
        vocab_size=project.vocab_size,
        context_length=project.context_length,
        d_model=project.d_model,
        n_heads=project.n_heads,
        n_layers=project.n_layers,
        dropout=project.dropout,
    )
    return GPT(model_config), project
