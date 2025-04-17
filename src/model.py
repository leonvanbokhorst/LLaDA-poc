import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class DiffusionConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        mask_token_id: int = 103,  # default BERT mask token
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.mask_token_id = mask_token_id


class DiffusionTransformer(nn.Module):
    """
    A simple mask-predictor Transformer for diffusion-based text generation.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            is_decoder=False,
            add_cross_attention=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            # Remove causal mask: BertModel is bidirectional by default
        )
        self.bert = BertModel(bert_config)
        self.mask_token_id = config.mask_token_id
        self.lm_head = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None
    ):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        returns logits: [batch_size, seq_len, vocab_size]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits
