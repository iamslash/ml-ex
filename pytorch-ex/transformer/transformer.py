import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # batch size
        N = query.shape[0]
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)    # (N, value_len, embed_size)
        keys = self.keys(keys)          # (N, key_len, embed_size)
        queries = self.queries(query)   # (N, query_len, embed_size)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, foward_expansion):
        super(TransformerBlock, self).__init__()

    def forward(self, value, key, query, mask):
        return None
    
class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

    def forward(self, x, value, key, src_mask, dst_mask):
        return None
    
class Decoder(nn.Module):
    def __init__(
            self,
            dst_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()

    def forward(self, x, enc_out, src_mask, dst_mask):
        return None
        

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            dst_vocab_size,
            src_pad_idx,
            dst_pad_idx,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_length=100,
    ):
        super(Transformer, self).__init__()

        

    def forward(self, src, dst):
        return dst

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # x: [2, 9], y[2, 8]
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    dst = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    dst_pad_idx = 0
    src_vocab_size = 10
    dst_vocab_size = 10
    model = Transformer(src_vocab_size, dst_vocab_size, src_pad_idx, dst_pad_idx, device=device).to(device)
    # x: [2, 9], y[2, 7]
    out = model(x, dst[:, :-1])
    print(out.shape)    
    