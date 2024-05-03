import json
from parameters import context_length


def get_songs():
    with open("collect_lyrics/lyrics.json", "r") as f:
        return json.load(f)


def prepare_data(length_data):
    """ Prepares the data for the model, creates vocab, dataset and tokenizer"""
    data = get_songs()
    vocab = set()

    for songs in data:
        for words in songs:
            vocab.add(words)

    vocab_size = len(vocab)
    sorted_list = sorted(list(vocab))
    int_to_char = {}
    char_to_int = {}

    for i, char in enumerate(sorted_list):
        int_to_char[char] = i + 1
        char_to_int[i + 1] = char

    dataset = []
    for song in data:
        start_index = 7
        while start_index + length_data < len(song):
            dataset.append(song[start_index: start_index + length_data])
            start_index += length_data
        dataset.append(song[len(song) - length_data:])

    return dataset, vocab_size + 1, int_to_char, char_to_int


length_data = context_length
dataset, vocab_size, char_to_int, int_to_char = prepare_data(length_data)
