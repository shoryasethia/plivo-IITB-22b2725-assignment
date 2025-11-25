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
EMAIL           P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=0.255 R=1.000 F1=0.406
PHONE           P=0.254 R=0.775 F1=0.383

Macro-F1: 0.465

PII-only metrics: P=0.404 R=0.755 F1=0.526
Non-PII metrics: P=1.000 R=1.000 F1=1.000
```

### Latency Performance
```
Latency over 50 runs (batch_size=1):
  p50: 17.56 ms
  p95: 27.66 ms
```

## Model Configuration

- **Base Model**: distilbert-base-uncased
- **Max Length**: 128 tokens
- **Epochs**: 5
- **Batch Size**: 16
- **Learning Rate**: 3e-5
- **Dropout**: 0.1

## Key Improvements

1. **Optimized Hyperparameters**: Increased epochs to 5, batch size to 16, and reduced learning rate to 3e-5 for better convergence
2. **Reduced Max Length**: Changed from 256 to 128 tokens to reduce latency
3. **Enhanced Span Decoding**: Improved BIO to span conversion for more robust entity extraction
4. **Dropout Regularization**: Added 0.1 dropout to prevent overfitting

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
  --max_length 128 \
  --dropout 0.1
```

## Prediction

```bash
# Dev set
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --max_length 128

# Stress set
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json \
  --max_length 128

# Test set
python src/predict.py \
  --model_dir out \
  --input data/test.jsonl \
  --output out/test_pred.json \
  --max_length 128
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

## Entity Labels

- **PII Entities**: CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE
- **Non-PII Entities**: CITY, LOCATION
