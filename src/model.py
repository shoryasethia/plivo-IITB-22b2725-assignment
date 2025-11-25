import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from labels import LABEL2ID, ID2LABEL


class TokenClassifier(nn.Module):
    """Custom lightweight token classifier with reduced classification head for better latency."""
    
    def __init__(self, model_name: str, dropout: float = 0.15):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        
        # Base encoder (DistilBERT or any BERT-like)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = config.hidden_size
        
        # Lightweight classification head (reduces latency significantly)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),  # Slightly less dropout in second layer
            nn.Linear(hidden_size // 2, len(LABEL2ID))
        )
        
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID
    
    def forward(self, input_ids, attention_mask):
        """Forward pass returns logits for token classification."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def create_model(model_name: str, dropout: float = 0.15):
    """Create custom token classifier with lightweight head for better latency."""
    return TokenClassifier(model_name, dropout)
