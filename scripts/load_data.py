from nltk import word_tokenize
import torch
from typing import List
from collections import defaultdict


def preprocess_dataset(file_name: str, word2id: dict, max_seq_len: int=100) -> List[torch.Tensor]:
    data = []
    with open(file_name, encoding="utf-8") as infile:
        for line in infile:
            tokens = word_tokenize(line.lower())
            data.append(tokens)

    sents = []
    vocab = {'$unk$': 0}
    for tokens in data:
        inds = [word2id[token] if token in word2id else 0 for token in tokens]
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        sents.append(torch.tensor(inds[: max_seq_len]))
    return sents, vocab
