"""Fine-tune DistilBERT for intent classification (plain PyTorch training loop).

Run:  python -m finpilot.intent.train [--epochs 3] [--batch 32] [--max-len 64]
Runs on CPU in roughly 30-60 minutes; uses a GPU automatically if one is available.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from finpilot import config
from finpilot.intent.data import LABELS, label_ids, load_splits

BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = config.MODELS_DIR / "distilbert-intent"


def encode(tokenizer, texts, labels, max_len: int) -> list[dict]:
    """Tokenise WITHOUT padding. Batches are padded later, only to their longest sentence.
    (The first version padded everything to 64 tokens; sentences average 16, so ~75% of
    the compute was spent on padding — see JOURNEY.md step 4.)"""
    enc = tokenizer(list(texts), truncation=True, max_length=max_len)
    return [{"input_ids": ids, "label": int(y)} for ids, y in zip(enc["input_ids"], labels)]


def make_collate(tokenizer):
    def collate(batch: list[dict]):
        padded = tokenizer.pad({"input_ids": [b["input_ids"] for b in batch]}, return_tensors="pt")
        return padded["input_ids"], padded["attention_mask"], torch.tensor([b["label"] for b in batch])
    return collate


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for input_ids, mask, labels in loader:
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    logits, labels = np.concatenate(all_logits), np.concatenate(all_labels)
    return float((logits.argmax(1) == labels).mean()), logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4,
                        help="CPU threads; on hybrid laptop CPUs, physical cores is often fastest "
                             "(see scripts/bench_training.py)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_splits()
    ids = label_ids()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=ids,
    ).to(device)

    data = {name: encode(tokenizer, df.text, df.label.map(ids).to_numpy(), args.max_len)
            for name, df in splits.items() if name in ("train", "val")}
    collate = make_collate(tokenizer)
    train_loader = DataLoader(data["train"], batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(data["val"], batch_size=128, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    history, best_acc = [], -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (input_ids, mask, labels) in enumerate(train_loader, start=1):
            out = model(input_ids=input_ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.3f} "
                      f"({time.time() - start:.0f}s)", flush=True)
        val_acc, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val_accuracy": val_acc})
        print(f"epoch {epoch} done: val accuracy {val_acc:.4f}", flush=True)
        if val_acc > best_acc:  # keep the best epoch, chosen on VALIDATION (not test)
            best_acc = val_acc
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)

    (OUT_DIR / "training_history.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "minutes": (time.time() - start) / 60}, indent=2))
    print(f"best val accuracy {best_acc:.4f}; saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
