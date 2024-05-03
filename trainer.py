import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from model import MyGPT
from prepare_data import dataset, char_to_int, vocab_size
from parameters import context_length, model_dim, num_heads, num_blocks, learning_rate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SongDataset(Dataset):
    def __init__(self, songs_data, word_to_int):
        self.songs_data = songs_data
        self.word_to_int = word_to_int

    def __len__(self):
        return len(self.songs_data)

    def __getitem__(self, idx):
        song = self.songs_data[idx]
        tokens = my_tokenizer(song, self.word_to_int)
        return torch.tensor(tokens)


def my_tokenizer(song, word_to_int):
    return [word_to_int[word] for word in song]


song_dataset = SongDataset(dataset, char_to_int)
dataloader = DataLoader(song_dataset, batch_size=10, shuffle=True)


model = MyGPT(vocab_size, context_length, model_dim, num_heads, num_blocks).to(device)
model.load_state_dict(torch.load("mygpt_model.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, batch in enumerate(dataloader):
        context = batch.to(device)
        optimizer.zero_grad()
        output = model(context)
        target = context[:, 1:]
        output = output[:, :-1, :]

        loss = criterion(output.contiguous().view(-1, vocab_size), target.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = output.argmax(dim=-1)
        correct_predictions += (predictions == target).sum().item()
        total_predictions += target.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}, Accuracy: {accuracy}")

torch.save(model.state_dict(), "mygpt_model.pth")
