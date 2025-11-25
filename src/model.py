import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from labels import LABEL2ID, ID2LABEL


class TokenClassifier(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)

        # Base encoder (DistilBERT or any BERT-like)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = config.hidden_size

        # Light classification head (helps latency & precision)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, len(LABEL2ID))
        )

        self.id2label = ID2LABEL
        self.label2id = LABEL2ID

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def create_model(model_name: str):
    return TokenClassifier(model_name)
