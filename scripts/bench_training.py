"""Measure seconds per training step for different CPU thread counts and padding.
Used once to choose training settings on a laptop CPU (see JOURNEY.md step 5)."""
import time, sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from finpilot.intent.data import load_splits

texts = list(load_splits()["train"].text.sample(32 * 4, random_state=0))
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=80)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
labels = torch.zeros(32, dtype=torch.long)

def step_time(threads, padding):
    torch.set_num_threads(threads)
    batches = [tok(texts[i:i+32], padding=padding, max_length=64, truncation=True, return_tensors="pt")
               for i in range(0, len(texts), 32)]
    model(**batches[0], labels=labels).loss.backward(); opt.zero_grad()  # warm-up
    start = time.perf_counter()
    for b in batches[1:]:
        model(**b, labels=labels).loss.backward(); opt.step(); opt.zero_grad()
    return (time.perf_counter() - start) / (len(batches) - 1)

for padding in ["max_length", "longest"]:
    for threads in [2, 4, 6, 10, 12]:
        print(f"padding={padding:10s} threads={threads:2d}: {step_time(threads, padding):.2f} s/step", flush=True)
