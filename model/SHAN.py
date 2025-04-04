import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_, uniform_

# from recbole.model.abstract_recommender import SequentialRecommender
# from recbole.model.loss import BPRLoss


class SHAN(nn.Module):
    def __init__(self, hidden_size, item_num, user_num, seq_size, device, short_item_length=10):
        super(SHAN, self).__init__()

        # load the dataset information
        self.user_num = user_num
        self.item_num = item_num
        self.device = device
        self.seq_size = seq_size
        # self.INVERSE_ITEM_SEQ = config["INVERSE_ITEM_SEQ"]

        # load the parameter information
        self.hidden_size = hidden_size
        self.short_item_length = short_item_length # the length of the short session items
        assert (
            self.short_item_length <= seq_size
        ), "short_item_length can't longer than the max_seq_length"
        # self.reg_weight = config["reg_weight"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            num_embeddings = self.item_num + 1, 
            embedding_dim = self.hidden_size, 
            padding_idx=0
        )
        self.user_embedding = nn.Embedding(self.user_num + 1, self.hidden_size)

        self.long_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.hidden_size),
                a=-np.sqrt(3 / self.hidden_size),
                b=np.sqrt(3 / self.hidden_size),
            ),
            requires_grad=True,
        )
        self.long_short_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.hidden_size),
                a=-np.sqrt(3 / self.hidden_size),
                b=np.sqrt(3 / self.hidden_size),
            ),
            requires_grad=True,
        )

        self.relu = nn.ReLU()
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0.0, 0.01)
        elif isinstance(module, nn.Linear):
            uniform_(
                module.weight.data,
                -np.sqrt(3 / self.hidden_size),
                np.sqrt(3 / self.hidden_size),
            )
        elif isinstance(module, nn.Parameter):
            uniform_(
                module.data,
                -np.sqrt(3 / self.hidden_size),
                np.sqrt(3 / self.hidden_size),
            )
            print(module.data)

    def forward(self, seq_item, user):
        seq_item_embedding = self.item_embedding(seq_item)
        user_embedding = self.user_embedding(user)

        # get the mask
        # seq_item = torch.tensor(seq_item)
        # print(seq_item.size())
        mask = seq_item.eq(0)

        long_term_attention_based_pooling_layer = (
            self.long_term_attention_based_pooling_layer(
                seq_item_embedding, user_embedding, mask
            )
        )
        # batch_size * 1 * embedding_size

        short_item_embedding = seq_item_embedding[:, -self.short_item_length :, :]
        mask_long_short = mask[:, -self.short_item_length :]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(self.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat(
            [long_term_attention_based_pooling_layer, short_item_embedding], dim=1
        )
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = (
            self.long_and_short_term_attention_based_pooling_layer(
                long_short_item_embedding, user_embedding, mask_long_short
            )
        )
        # batch_size * embedding_size

        return self.s_fc(long_short_item_embedding)

    def forward_eval(self, seq_item, user):
        return self.forward(seq_item, user)
    
    def long_and_short_term_attention_based_pooling_layer(
        self, long_short_item_embedding, user_embedding, mask=None
    ):
        """

        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.relu(
            self.long_short_w(long_short_item_embedding) + self.long_short_b
        )
        long_short_item_embedding = torch.matmul(
            long_short_item_embedding, user_embedding.unsqueeze(2)
        ).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(
            long_short_item_embedding_value, long_short_item_embedding.unsqueeze(2)
        ).sum(dim=1)
        return long_short_item_embedding

    def long_term_attention_based_pooling_layer(
        self, seq_item_embedding, user_embedding, mask=None
    ):
        """

        get the long term purpose of user
        """
        seq_item_embedding_value = seq_item_embedding

        seq_item_embedding = self.relu(self.long_w(seq_item_embedding) + self.long_b)
        user_item_embedding = torch.matmul(
            seq_item_embedding, user_embedding.unsqueeze(2)
        ).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(
            seq_item_embedding_value, user_item_embedding.unsqueeze(2)
        ).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding