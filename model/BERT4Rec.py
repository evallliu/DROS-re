import torch
from torch import nn
from utils.utility import extract_axis_1


class BERT4Rec(nn.Module):
    def __init__(self, hidden_size, item_num, seq_size, dropout_rate, device, n_layers=1, n_heads=1, inner_size=256, 
                 mask_ratio=0.1):
        super(BERT4Rec, self).__init__()

        # Load parameters info from config
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size  # Same as embedding_size
        self.inner_size = inner_size  # Dimensionality of the feed-forward layer
        self.dropout_rate = dropout_rate ##为了偷懒两个dropout用一个~
        self.mask_ratio = mask_ratio #指定在训练中随机掩盖（mask）多少比例的序列，通常用于自监督学习。

        # Dataset info
        self.mask_token = item_num  # Special mask token index
        self.mask_item_length = int(self.mask_ratio * seq_size)
        self.item_num = item_num
        self.max_seq_length = seq_size

        # Define layers
        self.item_embedding = nn.Embedding(
            num_embeddings = self.item_num + 1, 
            embedding_dim = self.hidden_size, 
            padding_idx=0
        )  # Mask token added
        self.position_embedding = nn.Embedding(
            num_embeddings = self.max_seq_length, 
            embedding_dim = self.hidden_size
        )  # Positional embeddings
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.dropout_rate,
                batch_first=True
            ),
            num_layers=self.n_layers
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_ffn = nn.Linear(hidden_size, hidden_size)
        self.output_gelu = nn.GELU()
        self.ln2 = nn.LayerNorm(hidden_size)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.output_bias = nn.Parameter(torch.zeros(self.item_num))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, len_item_seq):
        # Position encoding
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Item embedding
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding ##[256,10,64]
        # print(input_emb.size())
        # Layer normalization and dropout
        input_emb = self.ln1(input_emb)
        input_emb = self.dropout(input_emb)##[256,10,64]
        # print(input_emb.size())

        # Transformer encoder
        trm_output = self.transformer_encoder(input_emb)
        # print(trm_output.size())

        # Feedforward output
        ffn_output = self.output_ffn(trm_output)
        # print(ffn_output.size())
        ffn_output = self.output_gelu(ffn_output)
        state_hidden = extract_axis_1(ffn_output, len_item_seq - 1)
        output = self.s_fc(self.ln2(state_hidden)).squeeze()
        # print(output.size())

        return output  # [B, L, H]

    def forward_eval(self, item_seq,len_item_seq):
        # Evaluation forward pass (same as forward, but with specific handling for evaluation if needed)
        return self.forward(item_seq,len_item_seq)

