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
We could use cosine similarity or take the dot product of two word embeddings
P(q|f) = exp(lambda*cosine(q, f))/sum(exp(lambda*cosine(q, f')))
in which f' takes values from the vocab of foreign doc
lambda is a scaling factor that decides the shape of the softmax distribution
"""

import torch, logging, argparse
import numpy as np
from typing import List, Tuple
from logger import logging_config
from load_data import preprocess_dataset, read_file, CLIRDataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel


def get_parser():
    parser = argparse.ArgumentParser(description='Calculate word similarity scores')
    parser.add_argument('--query_embedding_file', type=str, help='file containing query static embedding', default=None)
    parser.add_argument('--foreign_embedding_file', type=str, help='file containing foreign static embedding', default=None)
    parser.add_argument('--query_file', type=str, help='file containing query sentences/docs')
    parser.add_argument('--foreign_file', type=str, help='file containing foreign sentences/docs')
    parser.add_argument('--static_only', action='store_true', help='outputs scores using only static embeddings')
    parser.add_argument('--contextualized_only', action='store_true', help='outputs scores using only contextualized embeddings')
    return parser


def token_lens_to_idxs(token_lens):
    """ This function is taken from ONEIE implementation
    Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


class CrossLingualIR:
    def __init__(self):
        self.emb_dim = None
        self.query_embeddings = None
        self.query_word2id = {}
        self.query_id2word = {}
        self.tgt_embeddings = None
        self.tgt_word2id = {}
        self.tgt_id2word = {}
        self.query_sents_contextualized, self.query_sents_static = [], []
        self.tgt_sents_contextualized, self.tgt_sents_static = [], []
        self.scores_static, self.scores_contextualized = None, None
        self.probs_static, self.probs_contextualized = None, None

    def load_embedding(self, embedding_file: str, query: bool) -> (torch.Tensor, dict, dict):
        """
        load pretrained embeddings from a text file
        :param embedding_file: name of the embedding file
        :param query: True if it is query word embeddings
        """
        weights = []
        word2id = {"$unk$": 0}
        with open(embedding_file) as infile:
            for i, line in enumerate(infile):
                if i==0:
                    split = line.split()
                    assert len(split) == 2
                    self.emb_dim = int(split[1])
                    # add unknown word to weights
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

    def get_similarity_scores_static(self, query: List[int], tgt_docs: List[List[int]], lambda_: int=1) -> None:
        """
        Calculate scores from static embeddings and save them to a .pt file
        :param query: query input ids
        :param tgt_docs: target input ids
        :param lambda_: scaling factor that decides the shape of the softmax distribution
        """
        tgt_doc_size, query_doc_size = len(tgt_docs), len(query)
        tgt_max_doc_len = max(len(tokens) for tokens in tgt_docs)
        tgt_embeddings = torch.zeros((tgt_doc_size, tgt_max_doc_len, self.emb_dim))
        for i in range(tgt_doc_size):
            ids = tgt_docs[i]
            for j in range(len(ids)):
                id = ids[j]
                tgt_embeddings[i, j, :] = self.tgt_embeddings[id]

        query_max_doc_len = max(len(tokens) for tokens in query)
        query_embeddings = torch.zeros((query_doc_size, query_max_doc_len, self.emb_dim))
        for i in range(query_doc_size):
            ids = query[i]
            for j in range(len(ids)):
                id = ids[j]
                query_embeddings[i, j, :] = self.query_embeddings[id]

        scores = torch.zeros((query_doc_size, tgt_doc_size, query_max_doc_len, tgt_max_doc_len))
        for i in range(query_doc_size):
            query_embedding = query_embeddings[i]
            # you could also try cosine similarity here but I got similar results
            similarity = torch.matmul(query_embedding, tgt_embeddings.transpose(1,2))
            scores[i] = (lambda_ * similarity).softmax(dim=0)
        torch.save(scores, '../out/score_static.pt')
        self.scores_static = scores

    def get_embeddings_contextualized(self, dataset: CLIRDataset) -> torch.Tensor:
        """
        Compute average contextualized embeddings for multi-piece words
        """
        # encoded input sentences with XLMRoberta
        model = XLMRobertaModel.from_pretrained('xlm-roberta-base',
                                                output_hidden_states=True)
        all_input_ids = [input_ids.input_ids for input_ids in dataset.data]
        all_token_len = [token_len.token_len for token_len in dataset.data]
        all_input_ids = torch.tensor(all_input_ids)
        all_outputs = model(all_input_ids)
        embeddings = all_outputs[2][0]  # shape (batch_size, max_seq_len, emb_dim)

        doc_size, _, emb_size = embeddings.shape

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = token_lens_to_idxs(all_token_len)
        # reshape idxs from (batch_size, seq_len) to (batch_size, seq_len, embed_size)
        idxs = all_input_ids.new(idxs).unsqueeze(-1).expand(doc_size, -1, emb_size) + 1
        masks = embeddings.new(masks).unsqueeze(-1)
        outputs = torch.gather(embeddings, 1, idxs) * masks
        outputs = outputs.view(doc_size, token_num, token_len, emb_size)
        outputs = outputs.mean(dim=2)
        return outputs

    def get_similarity_scores_contextualized(self, query_dataset: CLIRDataset, tgt_dataset: CLIRDataset,
                                             lambda_: int = 1) -> torch.Tensor:
        """
        Compute scores with contextualized embeddings and save them to a .pt file
        """
        # encoded input sentences with XLMRoberta
        self.tgt_sents_contextualized = [t.tokens for t in tgt_dataset.data]
        tgt_embeddings = self.get_embeddings_contextualized(tgt_dataset)  # shape (batch_size, max_seq_len, emb_dim)
        tgt_doc_size, tgt_token_num, tgt_emb_size = tgt_embeddings.shape

        self.query_sents_contextualized = [t.tokens for t in query_dataset.data]
        query_embeddings = self.get_embeddings_contextualized(query_dataset)  # shape (batch_size, max_seq_len, emb_dim)
        query_doc_size, query_token_num, query_emb_size = query_embeddings.shape

        scores = torch.zeros((query_doc_size, tgt_doc_size, query_token_num, tgt_token_num))
        for i in range(query_doc_size):
            query_embedding = query_embeddings[i]
            similarity = torch.matmul(query_embedding, tgt_embeddings.transpose(1,2))
            scores[i] = (lambda_ * similarity).softmax(dim=0)
        torch.save(scores, '../out/score_contextualized.pt')
        self.scores_contextualized = scores
        return scores

    def build_model_contextualized(self) -> torch.Tensor:
        scores = torch.load('../out/score_contextualized.pt')
        tgt_token_num = [len(tokens) for tokens in self.tgt_sents_contextualized]
        max_tgt_token_num = max(tgt_token_num)
        masks = []
        for token_num in tgt_token_num:
            masks.append([1] * token_num + [0] * (max_tgt_token_num - token_num))
        scores = scores * torch.tensor(masks, dtype=torch.float).unsqueeze(0).unsqueeze(2)
        tgt_token_num = torch.tensor(tgt_token_num, dtype=torch.float)
        probs = (scores.sum(dim=-1).transpose(2, 1)/tgt_token_num).transpose(2, 1).prod(dim=-1)
        self.probs_contextualized = probs
        return probs

    def build_model_static(self, tgt_docs: List[List[int]]) -> torch.Tensor:
        scores = torch.load('../out/score_static.pt')
        tgt_token_num = [len(tokens) for tokens in tgt_docs]
        max_tgt_token_num = max(tgt_token_num)
        masks = []
        for token_num in tgt_token_num:
            masks.append([1] * token_num + [0] * (max_tgt_token_num - token_num))
        scores = scores * torch.tensor(masks, dtype=torch.float).unsqueeze(0).unsqueeze(2)
        tgt_token_num = torch.tensor(tgt_token_num, dtype=torch.float)
        probs = (scores.sum(dim=-1).transpose(2, 1) / tgt_token_num).transpose(2, 1).prod(dim=-1)
        self.probs_static = probs
        return probs

    def get_top_docs(self, k: int=5, static: bool=True) -> None:
        query_sents = self.query_sents_static if static else self.query_sents_contextualized
        tgt_sents = self.tgt_sents_static if static else self.tgt_sents_contextualized
        probs = self.probs_static if static else self.probs_contextualized
        topk = probs.topk(k)
        for qry_ind, (inds, ps) in enumerate(zip(topk.indices, topk.values)):
            qry_sent = ' '.join(query_sents[qry_ind])
            print(f'query: {qry_sent}')
            for ind, p in zip(inds, ps):
                sent = ' '.join(tgt_sents[ind])
                print(f'{sent}, {p.item()}')
            print()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    logging_config('.', 'scores', level=logging.INFO)

    tgt_embedding_file = args.foreign_embedding_file
    query_embedding_file = args.query_embedding_file
    query_file = args.query_file
    foreign_file = args.foreign_file

    clir = CrossLingualIR()

    if args.static_only:
        # get scores using MUSE
        # preprocessing data
        tgt_embeddings, tgt_word2id, tgt_id2word = clir.load_embedding(tgt_embedding_file, False)
        query_embeddings, query_word2id, query_id2word = clir.load_embedding(query_embedding_file, True)
        query_input_ids, query_sents = preprocess_dataset(query_file, query_word2id)
        tgt_input_ids, tgt_sents = preprocess_dataset(foreign_file, tgt_word2id)
        clir.tgt_sents_static, clir.query_sents_static = tgt_sents, query_sents
        clir.get_similarity_scores_static(query_input_ids, tgt_input_ids)
        probs = clir.build_model_static(tgt_sents)
        clir.get_top_docs()
    elif args.contextualized_only:
        # get scores using XLM-Roberta
        query_dataset = CLIRDataset(query_file)
        tgt_dataset = CLIRDataset(foreign_file)

        scores = clir.get_similarity_scores_contextualized(query_dataset, tgt_dataset)
        probs = clir.build_model_contextualized()
        clir.get_top_docs(static=False)
    else:
        # get scores using MUSE
        # preprocessing data
        tgt_embeddings, tgt_word2id, tgt_id2word = clir.load_embedding(tgt_embedding_file, False)
        query_embeddings, query_word2id, query_id2word = clir.load_embedding(query_embedding_file, True)
        query_input_ids, query_sents = preprocess_dataset(query_file, query_word2id)
        tgt_input_ids, tgt_sents = preprocess_dataset(foreign_file, tgt_word2id)
        clir.tgt_sents_static, clir.query_sents_static = tgt_sents, query_sents
        clir.get_similarity_scores_static(query_input_ids, tgt_input_ids)
        probs = clir.build_model_static(tgt_sents)
        clir.get_top_docs()
        # get scores using XLM-Roberta
        query_dataset = CLIRDataset(query_file)
        tgt_dataset = CLIRDataset(foreign_file)

        scores = clir.get_similarity_scores_contextualized(query_dataset, tgt_dataset)
        probs = clir.build_model_contextualized()
        clir.get_top_docs(static=False)