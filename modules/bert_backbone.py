import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertBackbone(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size  # typically 768

    def forward(self, questions):
        """
        questions: list of strings of length B.
        Returns tensor of shape [B, seq_length, hidden_dim]
        """
        encoded = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=40)
        # Move to the same device as the model if needed (handled externally in training)
        outputs = self.bert(**encoded)
        return outputs.last_hidden_state  # [B, seq_length, hidden_dim]
