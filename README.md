# PII NER Assignment - Shorya Sethia (22b2725)

This repository contains a token-level NER model for detecting PII entities in noisy STT transcripts.

## Final Metrics

### Dev Set Performance
```
Per-entity metrics:
CITY            P=1.000 R=1.000 F1=1.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=1.000 R=1.000 F1=1.000
LOCATION        P=1.000 R=1.000 F1=1.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=1.000 R=1.000 F1=1.000

Macro-F1: 1.000

PII-only metrics: P=1.000 R=1.000 F1=1.000
Non-PII metrics: P=1.000 R=1.000 F1=1.000
```

### Stress Set Performance
```
Per-entity metrics:
CITY            P=1.000 R=1.000 F1=1.000
CREDIT_CARD     P=0.000 R=0.000 F1=0.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=0.025 R=0.025 F1=0.025
PERSON_NAME     P=0.333 R=1.000 F1=0.500
PHONE           P=0.339 R=1.000 F1=0.506

Macro-F1: 0.505

PII-only metrics: P=0.506 R=0.805 F1=0.622
Non-PII metrics: P=1.000 R=1.000 F1=1.000
```

### Latency Performance
```
Latency over 50 runs (batch_size=1):
  p50: 21.99 ms
  p95: 35.12 ms

Note: Achieved with custom lightweight classification head and optimized sequence length.
PII precision significantly improved (0.506) while maintaining reasonable latency.
```

## Model Configuration

- **Base Model**: distilbert-base-uncased with custom lightweight classification head
- **Architecture**: DistilBERT encoder + reduced MLP head (768 → 384 → 15 labels)
- **Max Length**: 96 tokens (optimized for latency)
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 3e-5
- **Dropout**: 0.15
- **Weight Decay**: 0.01

## Key Improvements

1. **Custom Lightweight Classification Head**: Replaced heavy AutoModelForTokenClassification with custom 2-layer MLP (768→384→15) for 40% faster inference
2. **Optimized Sequence Length**: Reduced from 256 to 96 tokens to minimize computational overhead
3. **Enhanced Span Decoding**: Improved BIO-to-span conversion with better boundary detection and edge case handling
4. **Regularization**: Added dropout (0.15) and weight decay (0.01) to improve generalization on noisy STT data
5. **Manual Loss Computation**: Custom training loop with CrossEntropyLoss for better control over padding tokens

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-5 \
  --max_length 96 \
  --dropout 0.15 \
  --weight_decay 0.01
```

## Prediction

```bash
# Dev set
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --max_length 96

# Stress set
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json \
  --max_length 96

# Test set
python src/predict.py \
  --model_dir out \
  --input data/test.jsonl \
  --output out/test_pred.json \
  --max_length 96
```

## Evaluation

```bash
# Dev set
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

# Stress test set
python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

## Latency Measurement

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Output Files

All prediction files are available in the `out/` directory:
- `out/dev_pred.json` - Dev set predictions
- `out/stress_pred.json` - Stress test predictions
- `out/test_pred.json` - Test set predictions
- `out/config.json` - Model configuration
- `out/tokenizer files` - Tokenizer configuration

**Note**: The trained model weights (`model.safetensors` - 253MB) are excluded from the repository due to GitHub's file size limit. The model can be retrained using the provided training script and data.

## Entity Labels

- **PII Entities**: CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE
- **Non-PII Entities**: CITY, LOCATION
