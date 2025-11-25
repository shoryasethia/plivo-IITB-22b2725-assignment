import json
import argparse
import torch
from transformers import AutoTokenizer
from labels import ID2LABEL, label_is_pii
import os


# ----------------------------------------------------
# CLEANING & POST-PROCESSING
# ----------------------------------------------------

def clean_spans(spans, min_conf=0.50):
    """Remove low-confidence or invalid spans."""
    cleaned = []
    for s in spans:
        if s["score"] < min_conf:
            continue
        if s["end"] - s["start"] < 2:  # avoid 1-char spans
            continue
        cleaned.append(s)
    return cleaned


# ----------------------------------------------------
# BIO → SPAN CONVERSION
# ----------------------------------------------------

def bio_to_spans(offsets, pred_ids, probs):
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_scores = []

    for (start, end), lid, p in zip(offsets, pred_ids, probs):
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")

        if label == "O":
            if current_label is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": current_label,
                    "score": sum(current_scores) / len(current_scores)
                })
            current_label = None
            current_scores = []
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            # close previous
            if current_label is not None:
                spans.append({
                    "start": current_start,
                    "end": current_end,
                    "label": current_label,
                    "score": sum(current_scores) / len(current_scores)
                })
            # start new
            current_label = ent_type
            current_start = start
            current_end = end
            current_scores = [p]

        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                current_scores.append(p)
            else:
                # mismatch → start new span
                if current_label is not None:
                    spans.append({
                        "start": current_start,
                        "end": current_end,
                        "label": current_label,
                        "score": sum(current_scores) / len(current_scores)
                    })

                current_label = ent_type
                current_start = start
                current_end = end
                current_scores = [p]

    # close last
    if current_label is not None:
        spans.append({
            "start": current_start,
            "end": current_end,
            "label": current_label,
            "score": sum(current_scores) / len(current_scores)
        })

    return spans


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )

    from model import TokenClassifier       # import your updated model class
    model = TokenClassifier(args.model_dir)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin")))
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = logits[0]
                probs = torch.softmax(logits, dim=-1)

                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                max_probs = probs.max(dim=-1).values.cpu().tolist()

            spans = bio_to_spans(offsets, pred_ids, max_probs)
            spans = clean_spans(spans)

            ents = []
            for sp in spans:
                ents.append({
                    "start": sp["start"],
                    "end": sp["end"],
                    "label": sp["label"],
                    "pii": bool(label_is_pii(sp["label"])),
                })

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances → {args.output}")


if __name__ == "__main__":
    main()
