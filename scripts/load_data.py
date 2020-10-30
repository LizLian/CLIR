from nltk import word_tokenize
import torch
from typing import List
from transformers import XLMRobertaTokenizer
from torch.utils.data import Dataset
from collections import namedtuple

instance_fields = ['sentence', 'tokens', 'token_len', 'subwords', 'input_ids']
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))


def preprocess_dataset(file_name: str, word2id: dict, max_seq_len: int=100) -> List[torch.Tensor]:
    sents = []
    with open(file_name, encoding="utf-8") as infile:
        for line in infile:
            tokens = word_tokenize(line.lower())
            sents.append(tokens)

    input_ids = []
    for tokens in sents:
        inds = [word2id[token] if token in word2id else 0 for token in tokens]
        input_ids.append(torch.tensor(inds[: max_seq_len]))
    return input_ids, sents


def read_file(file: str) -> List[str]:
    sents = []
    with open(file, encoding="utf-8") as infile:
        for line in infile:
            sents.append(line)
    return sents


class CLIRDataset(Dataset):
    def __init__(self, path: str, max_length: int=50):
        """
        :param path: path to the data file
        :param max_length: max sequence length
        """
        self.path = path
        self.max_length = max_length
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: str):
        return self.data[item]

    def load_data(self):
        """
        load data from file
        """
        with open(self.path, encoding="utf-8") as infile:
            for line in infile:
                tokens = word_tokenize(line)
                sentence = line
                # encode input sentences with XLMRoberta
                tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
                token_len = []
                subwords = []
                for token in tokens:
                    pieces = tokenizer.tokenize(token)
                    token_len.append(len(pieces))
                    subwords.extend(pieces)
                input_ids = tokenizer.encode(subwords,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length=self.max_length)
                pad_num = self.max_length - len(input_ids)
                input_ids = input_ids + [0] * pad_num
                instance = Instance(
                    sentence=sentence,
                    tokens=tokens,
                    token_len=token_len,
                    subwords=subwords,
                    input_ids=input_ids
                )
                self.data.append(instance)
        return self.data
