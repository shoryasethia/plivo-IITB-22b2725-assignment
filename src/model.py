from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = 0.1):
    # Load config and set dropout
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    
    # Set dropout based on model type
    if hasattr(config, 'hidden_dropout_prob'):
        config.hidden_dropout_prob = dropout
    if hasattr(config, 'attention_probs_dropout_prob'):
        config.attention_probs_dropout_prob = dropout
    if hasattr(config, 'dropout'):
        config.dropout = dropout
    if hasattr(config, 'seq_classif_dropout'):
        config.seq_classif_dropout = dropout
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
