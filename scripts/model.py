"""
Paper: Neural-Network Lexical Translation for Cross-lingual IR from Text and Speech
probabilitic cross-lingual IR model
P(Doc is R|Q) - given a query Q, the probability of the document is relevant
P(Doc is R|Q) is proportional to P(Q|Doc is R)
P(Q|Doc is R) = P(q1,q2,...,qn|f1,f2,...,fn)
q1,q2,...,qn are independent of each other
f1,f2,...,fn are independent of each other
P(q1|f1)*P(q2|f2)*...*P(qn|fn)
when ranking relevant documents, the ranking should be not only based on translation,
but it should also consider semantic similarity
We could use cosine similarity or take the softmax of the cosine similarity
P(q|f) = exp(lambda*cosine(q, f))/sum(exp(lambda*cosine(q, f')))    f' takes values from the vocab of foreign doc
lambda is a scaling factor that decides the shape of distribution of the softmax
"""

import torch
import numpy as np
import logging
from typing import List, Tuple
from logger import logging_config
from load_data import preprocess_dataset
from collections import defaultdict

logging_config('.', 'train', level=logging.INFO)


class CrossLingualIR:
    def __init__(self):
        self.emb_dim = None
        self.query_embeddings = None
        self.query_word2id = {}
        self.query_id2word = {}
        self.tgt_embeddings = None
        self.tgt_word2id = {}
        self.tgt_id2word = {}

    def load_embedding(self, embedding_file: str, query: bool) -> (torch.Tensor, dict, dict):
        """
        load pretrained embeddings from a text file
        :param embedding_file: name of the embedding file
        :param query: True if it is query word embeddings
        :return:
        """
        weights = []
        word2id = {"$unk$": 0}
        with open(embedding_file) as infile:
            for i, line in enumerate(infile):
                if i==0:
                    split = line.split()
                    assert len(split) == 2
                    self.emb_dim = int(split[1])
                    # add unkown word to weights
                    unk = np.random.uniform(-0.1, 0.1, self.emb_dim)
                    # unk = torch.distributions.uniform.Uniform(-0.1, 0.1).sample((self.emb_dim,))
                    weights.append(unk[None])
                else:
                    word, vect = line.rstrip().split(" ", 1)
                    vect = np.fromstring(vect, sep=" ")
                    if word in word2id:
                        logging.info(f"word {word} found twice in {embedding_file} file")
                    else:
                        if not vect.shape[0] == self.emb_dim:
                            logging.info(f"Invalid dimension {vect.shape} for word {word} in line {i}")
                            continue
                        word2id[word] = len(word2id)
                        weights.append(vect[None])
        assert len(word2id) == len(weights)

        logging.info(f'Loaded {len(weights)} pre-trained word embeddings')

        # compute new vocabulary / embeddings
        id2word = {v: k for k, v in word2id.items()}
        embeddings = np.concatenate(weights, 0)
        embeddings = torch.from_numpy(embeddings).float()
        if query:
            self.query_embeddings, self.query_word2id, self.query_id2word = embeddings, word2id, id2word
        else:
            self.tgt_embeddings, self.tgt_word2id, self.tgt_id2word = embeddings, word2id, id2word
        return embeddings, word2id, id2word

    def build_model(self, query: List[int], tgt_docs: List[List[int]], tgt_vocab: dict, lambda_: int=1) -> torch.Tensor:
        tgt_embedded = torch.zeros((len(tgt_vocab), self.emb_dim))
        for token in tgt_vocab:
            tgt_embedded[tgt_vocab[token]] = self.tgt_embeddings[
                self.tgt_word2id[token]] if token in self.tgt_word2id else 0
        cos = torch.nn.CosineSimilarity()
        probs = torch.ones((len(query), len(tgt_docs)), dtype=torch.float)
        for j in range(len(query)):
            for token in query[j]:
                q_embedding = self.query_embeddings[token]
                # cosine similarity
                similarity = cos(torch.unsqueeze(q_embedding, 0), tgt_embedded)
                # dot product
                similarity = torch.mm(q_embedding.unsqueeze(0), tgt_embedded.transpose(1,0)).squeeze(dim=0)
                prob = (lambda_ * similarity).softmax(dim=0)

                for i in range(len(tgt_docs)):
                    tgt_doc_prob = 0
                    tgt_doc = tgt_docs[i]
                    for f in tgt_doc:
                        tgt_doc_prob += prob[tgt_vocab[self.tgt_id2word[f.item()]]]
                    tgt_doc_prob /= tgt_doc.shape[0]
                    probs[j, i] *= tgt_doc_prob
        return probs


if __name__ == '__main__':
    clir = CrossLingualIR()
    # tgt_embedding_file = '../embeddings/wiki.multi.ar.vec'
    tgt_embedding_file = '../embeddings/wiki.multi.en.vec'
    query_embedding_file = '../embeddings/wiki.multi.en.vec'
    tgt_embeddings, tgt_word2id, tgt_id2word = clir.load_embedding(tgt_embedding_file, False)
    query_embeddings, query_word2id, query_id2word = clir.load_embedding(query_embedding_file, True)
    query_file = '../data/queries'
    foreign_file = '../data/foreign_docs_en'
    query_sents, _ = preprocess_dataset(query_file, query_word2id)
    tgt_sents, tgt_vocab = preprocess_dataset(foreign_file, tgt_word2id)
    probs = clir.build_model(query_sents, tgt_sents, tgt_vocab)
    topk = probs.topk(k=5)
    for k, p in zip(topk.indices[0], topk.values[0]):
        sent = [clir.tgt_id2word[i.item()] for i in tgt_sents[k.item()]]
        sent = " ".join(sent)
        print(f'{sent}, {p.item()}')
