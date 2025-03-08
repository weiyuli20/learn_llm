import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader



class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids =[]
        self.target_ids = []
        
        #分词
        token_ids = tokenizer.encode(txt,allowed_special={'<|endoftext|>'})

        #滑动窗口 获得输入，输出对
        for i in range(0,len(token_ids)- max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]


def create_dataloader_v1(txt,batch_size =4,max_length=256,
                            stride=128,shuffle=True,drop_last=True,num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt,tokenizer,max_length,stride)

    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            drop_last = drop_last,
                            num_workers = num_workers)
    return dataloader


class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out, num_heads, context_length,dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads ==0, "d_out must be divisable by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.Q = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.K = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.V = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                              torch.triu(torch.ones(context_length,context_length),diagonal = 1))
    

    def forward(self,x):
        b, num_tokens, d_in = x.shape

        k = self.K(x)  #(batch_size, num_tokens, d_out)
        q = self.Q(x)
        v = self.V(x)

        k = k.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        q = q.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)

        attn_scores = q @ k.transpose(-2,-1)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool,-torch.inf)
        attn_weights = torch.softmax(attn_scores /(self.d_out ** 0.5),dim=-1)

        attn_weights = self.dropout(attn_weights)  #(batch_size, num_head, num_tokens, num_tokens)

        context_vec = (attn_weights @ v).transpose(1,2) #(batch_size, num_tokens, num_head, head_dim)

        #view操作要求张量在内存中是连续的，否则会抛出错误。
        # 比如，如果原来的context_vec是通过某些操作（如transpose、permute、narrow等）得到的，那么它可能不是连续的。
        # 这时候直接调用view就会报错，提示张量不连续，需要使用contiguous()来确保内存布局连续。
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim = True)
        var = x.var(dim=-1,keepdim=True, unbiased = False)
        norm_x = (x-mean) / torch.sqrt(var +self.eps)

        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1+torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x+ 0.044715 * torch.pow(x,3))
        ))


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 *cfg["emb_dim"],cfg["emb_dim"])
        )

    def forward(self,x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size, seqlen = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seqlen,device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

def generate_text_simple(model,idx,max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        #crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits  = logits[:,-1,:]

        idx_next = torch.argmax(logits, dim = -1, keepdim = True)

        idx = torch.cat((idx,idx_next),dim=-1)

    return idx

def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:",start_context)
    print("Encoded input text:", encoded)
    print("Encoded_tensor.shape:",encoded_tensor.shape)

    out = generate_text_simple(
        model = model,
        idx = encoded_tensor,
        max_new_tokens =10,
        context_size = GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

if __name__ == '__main__':
    main()



