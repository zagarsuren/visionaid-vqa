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
        Returns tensor of shape [B, 1, hidden_dim] using only the [CLS] token.
        """
        encoded = self.tokenizer(
            questions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=40
        )
        # Move tensors to the same device as the model.
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        outputs = self.bert(**encoded)
        # Extract the [CLS] token embedding (first token) and add a token dimension.
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        return cls_embedding.unsqueeze(1)  # [B, 1, hidden_dim]
