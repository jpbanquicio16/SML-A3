# small_lm_rnn_vs_transformer_bpe.py
# -------------------------------------------------------------
# Train a small LM with (a) LSTM and (b) Transformer on the same
# dataset using a BPE tokenizer (vocab size = 10_000).
# -------------------------------------------------------------

import math
import os
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------
# Hyperparameters (shared)
# -------------------------
batch_size      = 64
block_size      = 256
max_iters       = 2000
eval_interval   = 250
eval_iters      = 100
learning_rate   = 3e-4
device          = "cuda" if torch.cuda.is_available() else "cpu"
dropout         = 0.2

# Transformer-specific
n_embd_trf  = 384
n_head      = 6
n_layer     = 6

# RNN-specific
n_embd_rnn  = 384
rnn_type    = "lstm"  # "lstm" or "gru"
rnn_layers  = 2

# Tokenizer
bpe_vocab_size     = 10_000
bpe_min_pair_freq  = 2  # ignore merges that appear <2 times to speed up
random_seed        = 1337

torch.manual_seed(random_seed)
print("Using device:", device)

# -------------------------
# Load data
# -------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------------------------------------------
# Byte-level BPE tokenizer (simple, self-contained)
#   - Starts from 256 single-byte tokens
#   - Trains merges up to target vocab size
#   - Stores merge ranks and token->bytes mapping
# -------------------------------------------------------------
class BPETokenizer:
    def __init__(self):
        self.byte_tokens = {i: (i,) for i in range(256)}  # id -> tuple(bytes)
        self.token2bytes = dict(self.byte_tokens)         # id -> tuple(bytes)
        self.bytes2token = {v: k for k, v in self.token2bytes.items()}  # tuple(bytes) -> id
        self.merges = {}          # (id_a, id_b) -> new_id
        self.rank = {}            # (id_a, id_b) -> merge order
        self.vocab_size = 256

    def _split_to_byte_ids(self, data_bytes):
        # returns a list[int] of base tokens (0..255)
        return list(data_bytes)

    def _count_adjacent_pairs(self, ids):
        counts = Counter()
        prev = ids[0]
        for cur in ids[1:]:
            counts[(prev, cur)] += 1
            prev = cur
        return counts

    def _merge_ids(self, ids, pair, new_id):
        a, b = pair
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, corpus_text, vocab_size=10_000, min_pair_freq=2):
        assert vocab_size >= 257
        data_bytes = corpus_text.encode("utf-8", errors="replace")
        ids = self._split_to_byte_ids(data_bytes)

        next_id = 256
        merge_order = 0
        print(f"Training BPE on {len(ids):,} byte tokens to vocab {vocab_size}…")
        t0 = time.time()

        while self.vocab_size < vocab_size:
            pair_counts = self._count_adjacent_pairs(ids)
            # filter low-frequency pairs for speed
            pair_counts = {p: c for p, c in pair_counts.items() if c >= min_pair_freq}
            if not pair_counts:
                print("No more pairs above min frequency. Stopping early.")
                break

            # pick most frequent pair
            best_pair, best_cnt = max(pair_counts.items(), key=lambda kv: kv[1])

            # create new token representing concatenation of bytes
            a, b = best_pair
            new_bytes = self.token2bytes[a] + self.token2bytes[b]
            self.token2bytes[next_id] = new_bytes
            self.bytes2token[new_bytes] = next_id
            self.merges[best_pair] = next_id
            self.rank[best_pair] = merge_order
            merge_order += 1

            # apply merge
            ids = self._merge_ids(ids, best_pair, next_id)

            next_id += 1
            self.vocab_size += 1

            if self.vocab_size % 500 == 0:
                elapsed = time.time() - t0
                print(f"  vocab size {self.vocab_size}, elapsed {elapsed:.1f}s")

        print(f"BPE training done: vocab={self.vocab_size}, time={time.time()-t0:.1f}s")

    # Greedy encoding: repeatedly merge best-ranked pairs present
    def encode(self, s):
        # start from raw bytes tokens
        ids = self._split_to_byte_ids(s.encode("utf-8", errors="replace"))
        if not self.rank:
            return ids
        # Convert to list of current token ids
        ids = list(ids)

        # Use a simple loop: try to merge while any applicable pair exists.
        # We recompute pair ranks each pass; good enough for small blocks.
        while True:
            best_rank = None
            best_pos = -1
            best_pair = None

            # scan pairs
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
                    best_pair = pair

            if best_rank is None:
                break  # no more merges applicable

            new_id = self.merges[best_pair]
            ids[best_pos:best_pos + 2] = [new_id]

        return ids

    def decode(self, ids):
        # expand each token id to bytes then join and decode
        out_bytes = bytearray()
        for t in ids:
            out_bytes.extend(self.token2bytes[t])
        return out_bytes.decode("utf-8", errors="replace")


# -------------------------
# Train tokenizer
# -------------------------
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=bpe_vocab_size, min_pair_freq=bpe_min_pair_freq)
vocab_size = tokenizer.vocab_size
print("Final vocab size:", vocab_size)

# -------------------------
# Numericalize dataset
# -------------------------
data_ids = tokenizer.encode(text)
data = torch.tensor(data_ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        lossv = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, l = model(X, Y)
            lossv[k] = l.item()
        losses[split] = lossv.mean().item()
    model.train()
    return losses

# -------------------------
# Models
# -------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=384, hidden_size=384, num_layers=2, dropout=0.2, rnn_type="lstm"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(n_embd, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.is_lstm = False
        else:
            self.rnn = nn.LSTM(n_embd, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.is_lstm = True
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None, hidden=None):
        x = self.embed(idx)
        x = self.dropout(x)
        if self.is_lstm:
            out, hidden = self.rnn(x, hidden)
        else:
            out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        logits = self.proj(out)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        hidden = None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, targets=None, hidden=hidden)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        self.train()
        return idx


class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f      = nn.LayerNorm(n_embd)
        self.head      = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# --------------------------------------
# Train/evaluate helpers for both models
# --------------------------------------
def train_model(model, label):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n=== Training {label} ===")
    t0 = time.time()
    for it in range(1, max_iters + 1):
        if it % eval_interval == 0 or it == 1 or it == max_iters:
            losses = estimate_loss(model)
            ppl = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
            print(f"[{label}] step {it:4d} | train {losses['train']:.3f} | val {losses['val']:.3f} | ppl {ppl:.2f}")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(f"Training time ({label}): {time.time() - t0:.1f}s")
    return model

def evaluate_model(model, label):
    losses = estimate_loss(model)
    ppl = math.exp(losses["val"]) if losses["val"] < 20 else float("inf")
    print(f"[{label}] Final: val loss {losses['val']:.4f} | perplexity {ppl:.2f}")
    return losses["val"], ppl

# -------------------------
# Train both models
# -------------------------
rnn_model = LSTMLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd_rnn,
    hidden_size=n_embd_rnn,
    num_layers=rnn_layers,
    dropout=dropout,
    rnn_type=rnn_type,
)
rnn_model = train_model(rnn_model, f"RNN({rnn_type.upper()})")

trf_model = TransformerLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd_trf,
    n_head=n_head,
    n_layer=n_layer,
)
trf_model = train_model(trf_model, "Transformer")

# -------------------------
# Evaluate and compare
# -------------------------
rnn_loss, rnn_ppl = evaluate_model(rnn_model, "RNN")
trf_loss, trf_ppl = evaluate_model(trf_model, "Transformer")

better = "Transformer" if trf_loss < rnn_loss else "RNN"
print("\n=== Comparison ===")
print(f"RNN:         val loss={rnn_loss:.4f}, perplexity={rnn_ppl:.2f}")
print(f"Transformer: val loss={trf_loss:.4f}, perplexity={trf_ppl:.2f}")
print(f"Winner (lower loss): {better}")

# -------------------------
# Short qualitative samples
# -------------------------
def sample_and_print(model, title, n_tokens=200):
    ctx = torch.zeros((1,1), dtype=torch.long, device=device)  # start with token 0 (some byte)
    out = model.generate(ctx, n_tokens)[0].tolist()
    txt = tokenizer.decode(out)
    print(f"\n--- {title} sample ---\n{txt[:1000]}\n")

sample_and_print(rnn_model, "RNN")
sample_and_print(trf_model, "Transformer")

# -------------------------
# Brief automatic analysis
# -------------------------
print("=== Brief Analysis ===")
if trf_loss < rnn_loss:
    print("On this dataset, the Transformer converged to a lower validation loss (and typically lower perplexity),")
    print("which suggests it models longer-range dependencies better than the RNN at similar parameter scales.")
    print("If the RNN underperforms, consider increasing its hidden size/layers or using LayerNorm/weight tying.")
else:
    print("On this dataset, the RNN matched or outperformed the Transformer; for smaller datasets or very short")
    print("contexts, RNNs can regularize implicitly and train faster. You can improve the Transformer by reducing")
    print("depth/head count, increasing dropout, or using weight decay/warmup.")
