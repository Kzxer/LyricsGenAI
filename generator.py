import torch
import torch.nn as nn
from model import MyGPT
from prepare_data import int_to_char, vocab_size
from parameters import context_length, model_dim, num_heads, num_blocks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, length_of_song, context, context_length, int_to_char):
    res = []
    for _ in range(length_of_song):
        if len(context.T) > context_length:
            context = context[:, -context_length:]
        prediction = model(context)
        last_prediction = prediction[:, -1:, :]
        probabilities = nn.functional.softmax(last_prediction, dim=-1)
        next_char = torch.multinomial(probabilities.squeeze(0), 1)
        context = torch.cat((context, next_char), dim=-1)
        res.append(int_to_char[next_char.item()])
    return ''.join(res)


model = MyGPT(vocab_size, context_length, model_dim, num_heads, num_blocks).to(device)
model.load_state_dict(torch.load("mygpt_model.pth"))
model.eval()
length_of_song = 128
context = torch.zeros(1, 1, dtype=torch.int64).to(device)


print(generate(model, length_of_song, context, context_length, int_to_char))

