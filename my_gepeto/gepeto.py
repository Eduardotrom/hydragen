import torch
import torch.nn as nn

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
from torch.nn import functional as F

# Params
block_size = 256  # context size
batch_size = 64
learning_rate = 3e-4
num_iters = 5000
max_new_tokens = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 20
max_iters = 100
n_embeddings = 384
n_head = 6
n_layer = 6
dropout = 0.2
filename = "input.txt"


with open(filename, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenize
# Convert tokens to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

n = int(0.9 * len(data))
train = data[:n]
validation = data[n:]
x = train[: block_size + 1]
y = train[1 : block_size + 1]

torch.manual_seed(12345)


def get_batch(split):
    data = train if split == "train" else validation
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch("train")


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        n_embd = n_embeddings
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BiGramLM(nn.Module):
    def __init__(self, vocab_size, n_layer):
        super().__init__()
        n_embd = n_embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size]
            logits, _loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx


m = BiGramLM(vocab_size, n_layer).to(device)
# idx = torch.zeros((1, 1), dtype=torch.long).to(device)
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for itera in range(max_iters):
    if itera % eval_iters == 0:
        losses = estimate_loss(m)
        print(
            f"step {itera}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(loss.item())
# print(
#     decode(
#         m.generate(
#             idx=torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100
#         )[0].tolist()
#     )
# )


# def attention_types():
#     # self attention
#     B, T, C = 4, 8, 2  # batch size,time,channel
#     x = torch.randn(B, T, C)
#     xbow = torch.zeros((B, T, C))

#     for b in range(B):
#         for t in range(T):
#             xprev = x[b, : t + 1]
#             xbow[b, t] = torch.mean(xprev, 0)

#     wei = torch.tril(torch.ones(T, T))
#     wei = wei / torch.sum(wei, 1, keepdim=True)
#     print(wei)
#     xbow2 = wei @ x  # T,T @ B,T,C -> B,T,C

#     tril = torch.tril(torch.ones(T, T))
#     wei = torch.zeros(T, T)
#     wei = wei.masked_fill(tril == 0, float("-inf"))
#     print(wei)
#     f = F.softmax(wei, dim=1)
#     xbow3 = wei @ x
