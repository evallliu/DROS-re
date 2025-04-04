import torch
from torch import nn
from torch.nn.init import normal_, xavier_normal_, constant_

'Caser is a model that incorporate CNN for recommendation.'

class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters_h, num_filters_v, filter_sizes,
                 dropout_rate):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters_h = num_filters_h
        self.num_filters_v = num_filters_v
        self.dropout_rate = dropout_rate

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1, 
                    out_channels=self.num_filters_h, 
                    kernel_size=(i, self.hidden_size)
                ) 
                for i in self.filter_sizes
            ]
        )

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(
            in_channels=1, out_channels=num_filters_v, kernel_size=(self.state_size, 1)
        )

        # Fully Connected Layer
        self.fc1_dim_v = self.num_filters_v * self.hidden_size
        self.fc1_dim_h = self.num_filters_h * len(self.filter_sizes)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.hidden_size)
        self.fc2 = nn.Linear(
            self.hidden_size, item_num
        )

        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 1.0 / module.embedding_dim)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, states, len_states):
        # 1. 获取物品序列的嵌入，并增加维度，使其适用于 2D 卷积
        input_emb = self.item_embeddings(states)  # (batch_size, seq_len, embedding_dim)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim) 用于 CNN

        # 2. 初始化输出变量
        out, h_out, v_out = None, None, None

        # 3. 处理垂直（vertical）方向的卷积（n_v 是控制是否进行该卷积的标志）
        if self.num_filters_v:
            v_out = self.vertical_cnn(input_emb)  # (batch_size, num_filters_v, 1, 1)
            v_flat = v_out.view(-1, self.fc1_dim_v)  # 展平数据，为全连接层准备 (batch_size, fc1_dim_v)

        # 4. 处理水平（horizontal）方向的卷积（n_h 是控制是否进行该卷积的标志）
        pooled_outputs = list()  # 用于存放多个卷积核处理后的结果
        if self.num_filters_h:
            for cnn in self.horizontal_cnn:  # 遍历多个卷积核
                h_out = nn.functional.relu(cnn(input_emb)) # (batch_size, num_filters_h, new_seq_len)?
                h_out = h_out.squeeze()  
                p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])  # (batch_size, num_filters_h)
                pooled_outputs.append(p_out)  # 存储每个卷积核的结果
            
            # 将多个卷积核的结果拼接 (batch_size, total_filters_h)
        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.fc1_dim_h)

        # 5. 将垂直和水平的卷积输出拼接成最终的特征
        out = torch.cat([v_flat, h_pool_flat], 1)  # (batch_size, fc1_dim_v + total_filters_h)

        # 6. 通过 Dropout 层防止过拟合
        out = self.dropout(out)

        # 7. 通过全连接层 + 激活函数
        z = nn.functional.relu(self.fc1(out))  # (batch_size, hidden_dim)

        # 8. 通过第二个全连接层，生成最终的序列输出
        supervised_output = self.fc2(z)  # (batch_size, output_dim)
        return supervised_output  # 输出最终的表示


    def forward_eval(self, states, len_states):
        return self.forward(states, len_states)
