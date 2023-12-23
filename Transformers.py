import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConfig:
    def __init__(self, 
                 emb_size=512, 
                 n_heads=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6, 
                 dropout=0.1):
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, n_heads):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.head_dim = emb_size // n_heads

        assert (
            self.head_dim * n_heads == emb_size
        ), "Embedding size must be divisible by number of heads"

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]
        seq_length = q.shape[1]

        qkv = self.qkv(torch.cat((q, k, v), dim=0)).view(batch_size, seq_length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Escalar produto de atenção
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(batch_size, seq_length, self.emb_size)
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_size, expansion=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, emb_size, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, n_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(attn + x)
        x = self.dropout(x)
        ff = self.ff(x)
        x = self.norm2(ff + x)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, emb_size, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(emb_size, n_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.transformer_block = MultiHeadAttention(emb_size, n_heads)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = FeedForward(emb_size, dropout=dropout)
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        attn = self.attention(x, x, x, trg_mask)
        x = self.norm1(attn + x)
        x = self.dropout(x)
        transformer_block = self.transformer_block(x, enc_out, enc_out, src_mask)
        x = self.norm2(transformer_block + x)
        x = self.dropout(x)
        ff = self.ff(x)
        x = self.norm3(ff + x)
        x = self.dropout(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, config):
        super().__init__()

        self.encoder = Encoder(src_vocab_size, config)
        self.decoder = Decoder(trg_vocab_size, config)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, config.emb_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(config.emb_size, config.n_heads, config.dropout) for _ in range(config.num_encoder_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, config.emb_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config.emb_size, config.n_heads, config.dropout) for _ in range(config.num_decoder_layers)]
        )
        self.fc_out = nn.Linear(config.emb_size, vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

# Defina os tamanhos do vocabulário e os índices de padding
src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = 0
trg_pad_idx = 0

# Crie a configuração e o modelo do Transformer
config = TransformerConfig()
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, config)

# Teste com um exemplo simples
src = torch.randint(0, src_vocab_size, (1, 10))
trg = torch.randint(0, trg_vocab_size, (1, 10))

with torch.no_grad():
    out = model(src, trg)

print(out)
