import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim=512, num_heads=8):
        super().__init__()
        # Project visual and textual features into the same hidden space.
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # Cross-modal attention: use visual features as queries and text as key/value.
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_feat, text_feat):
        """
        Args:
            visual_feat: Tensor of shape [B, N, visual_dim] (visual tokens).
            text_feat: Tensor of shape [B, M, text_dim] (textual tokens).
        Returns:
            fused_feat: Tensor of shape [B, N, hidden_dim] – refined visual features.
            attn_weights: Attention weights for interpretability.
        """
        v_proj = self.visual_proj(visual_feat)  # [B, N, hidden_dim]
        t_proj = self.text_proj(text_feat)        # [B, M, hidden_dim]
        attn_output, attn_weights = self.cross_attention(query=v_proj, key=t_proj, value=t_proj)
        fused_feat = self.norm(v_proj + attn_output)
        return fused_feat, attn_weights

class MultimodalVQAWithAttention(nn.Module):
    def __init__(self, visual_backbone, text_backbone, num_answers):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.text_backbone = text_backbone
        self.classifier = nn.Linear(visual_backbone.output_dim + text_backbone.output_dim, num_answers)

    def forward(self, image, question, label_ids=None):
        """
        Forward pass for the multimodal VQA model.
        Args:
            image: Tensor of shape [B, 3, H, W] (image features).
            question: Tokenized question input for the text backbone.
            label_ids: (Optional) Ground truth labels, not used in forward pass.
        Returns:
            logits: Tensor of shape [B, num_answers].
            attn_weights: Attention weights (if applicable).
        """
        visual_features = self.visual_backbone(image)  # Shape: [B, 1, 2048]
        text_features = self.text_backbone(question)  # Shape: [B, 1, text_backbone.output_dim]
        combined_features = torch.cat((visual_features, text_features), dim=-1)  # Shape: [B, 1, 2048 + text_backbone.output_dim]
        logits = self.classifier(combined_features.squeeze(1))  # Shape: [B, num_answers]
        return logits, None  # Replace None with attention weights if applicable
