import json
import argparse
import torch
from transformers import AutoTokenizer
from labels import ID2LABEL, label_is_pii
from model import create_model
import os


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # Skip special tokens (CLS, SEP, PAD)
        if start == 0 and end == 0:
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            # End current entity if exists
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None
            continue

        # Parse BIO tag
        if "-" not in label:
            continue
            
        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            # Save previous entity if exists
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            # Start new entity
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            # Continue entity only if same type
            if current_label == ent_type:
                current_end = end
            else:
                # Treat as beginning of new entity if type mismatch
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    # Add final entity if exists
    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=96)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    
    # Load custom model
    model = create_model(args.model_dir if args.model_name is None else args.model_name)
    
    # Load weights
    model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    
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
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_ids = logits[0].argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
