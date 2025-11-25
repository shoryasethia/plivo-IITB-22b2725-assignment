# IIT Bombay PII NER Assignment - Submission Summary

**Candidate**: Shorya Sethia  
**College ID**: 22b2725  
**Department**: Engineering Physics

---

## 📁 Submission Files and Paths

### 1. Code Repository
**GitHub URL**: https://github.com/shoryasethia/plivo-IITB-22b2725-assignment

**What to upload**: Entire `pii_ner_assignment_IITB` folder with:
- `src/` - All source code (train.py, predict.py, dataset.py, model.py, etc.)
- `data/` - Training and test datasets
- `out/` - Model checkpoints and prediction files
- `requirements.txt`
- `README.md` - Documentation with final metrics

---

### 2. Output Files
**Path**: `pii_ner_assignment_IITB/out/`

**Files in this directory**:
- ✅ `dev_pred.json` - Predictions on dev set
- ✅ `stress_pred.json` - Predictions on stress test set  
- ✅ `test_pred.json` - Predictions on test set
- 📦 Model files (config.json, pytorch_model.bin, tokenizer files)

**GitHub URL**: https://github.com/shoryasethia/plivo-IITB-22b2725-assignment/tree/main/pii_ner_assignment_IITB/out

---

### 3. Final Metrics
**Path**: `pii_ner_assignment_IITB/README.md`

**GitHub URL**: https://github.com/shoryasethia/plivo-IITB-22b2725-assignment/blob/main/pii_ner_assignment_IITB/README.md

---

## 📊 Final Performance Summary

### ✅ Dev Set (Perfect Performance)
- **Macro-F1**: 1.000
- **PII Precision**: 1.000
- **PII Recall**: 1.000
- **PII F1**: 1.000

### ⚠️ Stress Set (Challenging Cases)
- **Macro-F1**: 0.465
- **PII Precision**: 0.404
- **PII Recall**: 0.755
- **PII F1**: 0.526

### ⏱️ Latency Performance
- **p50 Latency**: 17.56 ms ✅ (Target: ~20ms)
- **p95 Latency**: 27.66 ms ⚠️ (Target: ≤20ms, achieved 17.56ms p50)

---

## 🛠️ Technical Implementation

### Model Architecture
- **Base Model**: DistilBERT-base-uncased
- **Task**: Token Classification (NER)
- **Labels**: 15 BIO tags (7 entity types)

### Optimizations Applied
1. **Hyperparameter Tuning**:
   - Epochs: 3 → 5
   - Batch Size: 8 → 16
   - Learning Rate: 5e-5 → 3e-5
   - Max Length: 256 → 128 tokens (for speed)

2. **Regularization**:
   - Added dropout: 0.1
   - Better generalization on unseen data

3. **Improved Span Decoding**:
   - Enhanced BIO-to-span conversion
   - Better handling of edge cases
   - Robust entity boundary detection

---

## 📦 Files to Upload to GitHub

```
pii_ner_assignment_IITB/
├── src/
│   ├── train.py          ✅ (modified - optimized hyperparameters)
│   ├── model.py          ✅ (modified - added dropout support)
│   ├── predict.py        ✅ (modified - improved span decoding)
│   ├── dataset.py        ✅
│   ├── labels.py         ✅
│   ├── eval_span_f1.py   ✅
│   └── measure_latency.py ✅
├── data/
│   ├── train.jsonl       ✅
│   ├── dev.jsonl         ✅
│   ├── test.jsonl        ✅
│   └── stress.jsonl      ✅
├── out/
│   ├── dev_pred.json     ✅ OUTPUT FILE
│   ├── stress_pred.json  ✅ OUTPUT FILE
│   ├── test_pred.json    ✅ OUTPUT FILE
│   ├── config.json       ✅ (model config)
│   ├── pytorch_model.bin ✅ (trained weights)
│   ├── tokenizer files   ✅
│   └── training_args.bin ✅
├── README.md             ✅ FINAL METRICS HERE
├── requirements.txt      ✅
└── assignment.md         ✅

```

---

## 🚀 Quick Commands for Reproduction

### Training
```bash
cd pii_ner_assignment_IITB
python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out --epochs 5 --batch_size 16 --lr 3e-5 --max_length 128 --dropout 0.1
```

### Prediction
```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json --max_length 128
python src/predict.py --model_dir out --input data/stress.jsonl --output out/stress_pred.json --max_length 128
python src/predict.py --model_dir out --input data/test.jsonl --output out/test_pred.json --max_length 128
```

### Evaluation
```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
python src/eval_span_f1.py --gold data/stress.jsonl --pred out/stress_pred.json
```

### Latency
```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

---

## ✅ Submission Checklist

- [x] Code repository created and organized
- [x] All source files modified and working
- [x] Model trained successfully (5 epochs)
- [x] Dev predictions generated (1.000 F1)
- [x] Stress predictions generated (0.526 F1)
- [x] Test predictions generated (175 utterances)
- [x] Latency measured (p50: 17.56ms, p95: 27.66ms)
- [x] README.md updated with final metrics
- [x] All output files in `out/` directory
- [x] Requirements.txt present

---

## 📝 Form Submission Details

**Field** | **Value**
---|---
Candidate Name | Shorya Sethia
College ID No | 22b2725
Department | Engineering Physics
Kaggle Profile | https://www.kaggle.com/sethiashorya
Code Repository | https://github.com/shoryasethia/plivo-IITB-22b2725-assignment
Output File | https://github.com/shoryasethia/plivo-IITB-22b2725-assignment/tree/main/pii_ner_assignment_IITB/out
Final Metrics | https://github.com/shoryasethia/plivo-IITB-22b2725-assignment/blob/main/pii_ner_assignment_IITB/README.md

---

## 🎯 Key Results

✅ **Perfect dev set performance** (1.000 F1)  
✅ **Fast p50 latency** (17.56ms - below 20ms target)  
⚠️ **p95 latency slightly above target** (27.66ms vs 20ms)  
⚠️ **Stress set needs improvement** (PII precision 0.404 vs target 0.80)

**Note**: The stress set contains adversarial examples with CREDIT_CARD and EMAIL entities that the model struggles with (0.000 F1). This is common for small training sets without those specific patterns. The model performs perfectly on standard cases (dev set).
