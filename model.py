import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyGPT(nn.Module):
    class TransformerBlock(nn.Module):
        class MultiHeadAttention(nn.Module):
            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim, head_size):
                    super().__init__()
                    self.key = nn.Linear(model_dim, head_size, bias=False)
                    self.query = nn.Linear(model_dim, head_size, bias=False)
                    self.value = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.key(embedded)
                    q = self.query(embedded)
                    v = self.value(embedded)

                    scores = q @ torch.transpose(k, 1, 2)
                    context_len, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    pre_mask = torch.tril(torch.ones(context_len, context_len))
                    mask = (pre_mask == 0).to(device)
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2)

                    return scores @ v

            def __init__(self, model_dim, num_heads):
                super().__init__()
                self.attention_heads = nn.ModuleList()
                for _ in range(num_heads):
                    self.attention_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.linear = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, embedded):
                output = []
                for head in self.attention_heads:
                    output.append(head(embedded))
                concatenated = torch.cat(output, dim=2)
                return self.dropout(self.linear(concatenated))

        class FeedForward(nn.Module):
            def __init__(self, model_dim):
                super().__init__()
                self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))

        def __init__(self, model_dim, num_heads):
            super().__init__()
            self.multi_attention = self.MultiHeadAttention(model_dim, num_heads)
            self.feed_forward = self.FeedForward(model_dim)
            self.layer_norm_1 = nn.LayerNorm(model_dim)
            self.layer_norm_2 = nn.LayerNorm(model_dim)

        def forward(self, embedded):
            embedded = embedded + self.multi_attention(self.layer_norm_1(embedded))
            embedded = embedded + self.feed_forward(self.layer_norm_2(embedded))
            return embedded

    def __init__(self, vocab_size, context_length, model_dim, num_heads, num_blocks):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(context_length, model_dim)
        self.transformer = nn.Sequential()
        for _ in range(num_blocks):
            self.transformer.append(self.TransformerBlock(model_dim, num_heads))
        self.final_layer_norm = nn.LayerNorm(model_dim)
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        embedded = self.token_embedding(context)
        context_len = embedded.shape[1]
        position = torch.arange(context_len).to(device)
        embedded = embedded + self.position_embedding(position)
        raw_output = self.output_layer(self.final_layer_norm(self.transformer(embedded)))
        return raw_output

