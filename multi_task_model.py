import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes_task_a: int = 3,
        num_classes_task_b: int = 2,
        embedding_dim: int = 768,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        # Transformer backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Pooling layer for sentence embeddings
        self.pooling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )
        
        # Task-specific heads
        # Task A: Sentence Classification
        self.task_a_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes_task_a)
        )
        
        # Task B: Sentiment Analysis (example second task)
        self.task_b_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, num_classes_task_b)
        )
        
    def get_sentence_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract sentence embeddings from transformer outputs"""
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token and apply pooling
        cls_embedding = outputs.last_hidden_state[:, 0]
        sentence_embedding = self.pooling(cls_embedding)
        
        return sentence_embedding
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional task specification
        task: Either 'task_a', 'task_b', or None (returns embeddings only)
        """
        sentence_embedding = self.get_sentence_embedding(input_ids, attention_mask)
        outputs = {'embeddings': sentence_embedding}
        if task == 'task_a':
            outputs['task_a_logits'] = self.task_a_head(sentence_embedding)
        elif task == 'task_b':
            outputs['task_b_logits'] = self.task_b_head(sentence_embedding)
        elif task is None:
            outputs['task_a_logits'] = self.task_a_head(sentence_embedding)
            outputs['task_b_logits'] = self.task_b_head(sentence_embedding)
        return outputs

class LayerwiseLearningRateOptimizer:
    def __init__(
        self,
        model: MultiTaskSentenceTransformer,
        lr_backbone: float = 1e-5,
        lr_pooling: float = 2e-5,
        lr_heads: float = 5e-5
    ):
        # Group parameters by component
        backbone_params = self.backbone.named_parameters()
        pooling_params = self.pooling.named_parameters()
        task_heads_params = list(self.task_a_head.named_parameters()) + \
                           list(self.task_b_head.named_parameters())
        
        # Create parameter groups with different learning rates
        self.param_groups = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': pooling_params, 'lr': lr_pooling},
            {'params': task_heads_params, 'lr': lr_heads}
        ]
        
        self.optimizer = torch.optim.AdamW(self.param_groups)

if __name__=="__main__":
    model = MultiTaskSentenceTransformer()
