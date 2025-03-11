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
        """
        Args:
            visual_backbone: a module that outputs visual features of shape [B, N, visual_dim]
            text_backbone: a module that outputs text embeddings of shape [B, M, text_dim]
            num_answers: number of answer classes
        """
        super().__init__()
        self.visual_backbone = visual_backbone  # e.g., a CNN or ViT that produces patch features
        self.text_backbone = text_backbone      # e.g., BERT (or its embedding part)
        self.attention_module = CrossModalAttention(
            visual_dim=visual_backbone.output_dim,
            text_dim=text_backbone.output_dim,
            hidden_dim=512,
            num_heads=8
        )
        # After fusion, pool the tokens and pass through a classifier.
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over the token dimension
        self.classifier = nn.Linear(512, num_answers)

    def forward(self, image, question):
        """
        Args:
            image: tensor of shape [B, 3, H, W]
            question: list of strings (length B)
        Returns:
            logits: tensor of shape [B, num_answers]
            attn_weights: cross-modal attention weights
        """
        visual_feat = self.visual_backbone(image)  # [B, N, visual_dim]
        text_feat = self.text_backbone(question)     # [B, M, text_dim]
        fused_feat, attn_weights = self.attention_module(visual_feat, text_feat)  # [B, N, 512]
        # Pool fused tokens along the token dimension.
        fused_feat = fused_feat.transpose(1, 2)  # [B, 512, N]
        pooled_feat = self.pool(fused_feat).squeeze(-1)  # [B, 512]
        logits = self.classifier(pooled_feat)  # [B, num_answers]
        return logits, attn_weights
