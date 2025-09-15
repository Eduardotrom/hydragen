import torch
import torch.nn as nn
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
from torch.nn import functional as F
filename="input.txt"
with open(filename,"r",encoding="utf-8") as file:
    text=file.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)

#tokenize
#Convert tokens to integers
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:"".join(itos[i] for i in l)

data=torch.tensor.encode(text,dtype=torch.long)

n=int(0.9*len(data))
train=data[:n]
validation=data[n:]
block_size=8 #context size
x=train[:block_size+1]
y=train[1:block_size+1]

torch.manual.seed(12345)
batch_size=4
block_size=8


def get_batch(split):
    data=train if split=="train" else validation
    ix=torch.randint(len(data))-block_size,(batch_size,)
    x=torch.stack(data[i:i+block_size] for i in ix)
    y=torch.stack(data[i+1:i+block_size+1] for i in ix)
    return x,y

xb,yb=get_batch("train")

class BiGramLM(nn.Module):
    def __init__(self, vocab_size):
        self.super__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
            logits = self.token_embedding_table(idx)
            B,T,C=logits.shape()
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=0
            if targets is not None:
                loss=F.cross_entropy(logits,targets)
            return logits,loss
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=1)
            id_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,id_next),dim=1)
            return idx



m=BiGramLM(vocab_size)
out=m(xb,yb)
idx=torch.zeros((1,1),dtype=torch.long)
print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))
optimizer=torch.optimize.Adamw(m.parameters(),lr=1e-3)

batch_size=32
for steps in range(100):
     xb,yb=get_batch("train")
     logits,loss=m(xb,yb)
     optimizer.zero_grad(set_to_none=True)
     loss.backward()
     optimizer.step()


print(loss.item())



#self attention
B,T,C=4,8,2#batch size,time,channel
x=torch.randn(B,T,C)
xbow=torch.zeros((B,T,C))

for b in range(B):
     for t in range(T):
          xprev=x[b:t+1]
          xbow[b,t]=torch.mean(xprev,0)

wei=torch.tril(torch.ones(T,T))
we=wei/wei(sum(1,keepdim=True))
xbow2=wei@x

tril=0
wei=torch.zeros(T,T)
wei=wei.masked_fill(tril==0,float("-inf"))
f=F.softmax(wei,dim=1)
xbow3=we@x