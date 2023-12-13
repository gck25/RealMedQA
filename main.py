import os
import json
import faiss
import torch
import argparse
import gensim
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import ndcg_score, average_precision_score

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def create_dataset_tokenized(dataset, tokenizer, device):
    dataset_tokenized = tokenizer(dataset, padding=True, truncation=True)

    dataset_tokenized = {
        'input_ids': torch.tensor(dataset_tokenized['input_ids']).to(device),
        'token_type_ids': torch.tensor(dataset_tokenized['token_type_ids']).to(device),
        'attention_mask': torch.tensor(dataset_tokenized['attention_mask']).to(device),
    }

    return dataset_tokenized


def calculate_metrics_llm(questions_embeddings, index):
    N = len(questions_embeddings)

    recall_dict = {}
    random_dict = {}
    ncdg_dict = {}
    map_dict = {}
    for k in [1, 2, 5, 10, 20, 50, 100]:
        random_dict[f"random@{k}"] = k / N
        recall = 0
        ncdg_total = 0
        map_ = 0
        for i in range(N):
            A, I = index.search(questions_embeddings[i:i + 1], k)

            if i in I:
                recall += 1
                if k > 1:
                    y_true = np.array([0.0] * A)
                    ind = list(I[0]).index(i)
                    y_true[0][ind] = 1.0
                    ncdg_total += ndcg_score(y_true=y_true, y_score=A, k=k)
                    map_ += average_precision_score(y_true=y_true[0], y_score=A[0])
        if k > 1:
            ncdg_total /= N
            ncdg_dict[f"ncdg@{k}"] = ncdg_total
            map_ /= N
            map_dict[f"map@{k}"] = map_
        recall /= N
        recall_dict[f"recall@{k}"] = recall

    return recall_dict, ncdg_dict, map_dict


def calculate_metrics_bm25(questions, tokenized_questions, answers, answers_unique):
    N = len(questions)
    recall_dict = {}
    ncdg_dict = {}
    map_dict = {}
    for k in [1, 2, 5, 10, 20, 50, 100]:
        recall = 0
        ncdg_total = 0
        map_ = 0
        for q, tq, ans in zip(questions, tokenized_questions, answers):
            top_k = bm25.get_top_n(tq, answers_unique, n=k)
            if ans in top_k:
                recall += 1
                if k > 1:
                    ind = top_k.index(ans)
                    scores = bm25.get_scores(tq)

                    y_score = [scores[i] for i, d in enumerate(answers_unique) if d in top_k]
                    norm = sum(y_score)
                    y_score = np.array([score / norm for score in y_score]).reshape(1, -1)

                    y_true = np.array([0] * k).reshape(1, -1)

                    y_true[0][ind] = 1
                    ncdg_total += ndcg_score(y_true=y_true, y_score=y_score, k=k)
                    map_ += average_precision_score(y_true=y_true[0], y_score=y_score[0], pos_label=1)

        recall /= N
        recall_dict[f"recall@{k}"] = recall
        ncdg_total /= N
        ncdg_dict[f"ncdg@{k}"] = ncdg_total
        map_ /= N
        map_dict[f"map@{k}"] = map_
        return recall_dict, ncdg_dict, map_dict

def main(args):
    model_type = args.model_type
    dataset_type = args.dataset_type
    batch_size = args.batch_size

    if dataset_type == "BioASQ":
        questions = None
        answers = None
    else:
        questions = None
        answers = None

    if model_type == "bm25":
        answers_unique = list(set(answers))
        tokenized_answers = [list(gensim.utils.tokenize(doc.lower())) for doc in answers_unique]
        tokenized_questions = [list(gensim.utils.tokenize(q.lower())) for q in questions]
        bm25 = BM25Okapi(tokenized_answers)
        recall_dict, ncdg_dict, map_dict = calculate_metrics_bm25(bm25, questions, tokenized_questions, answers,
                                                                  answers_unique)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModel.from_pretrained(model_type).to(device)


        questions_tokenized = create_dataset_tokenized(questions, tokenizer, device, batch_size)
        answers_tokenized = create_dataset_tokenized(questions, tokenizer, device, batch_size)

        aloader = DataLoader(list(zip(answers_tokenized['input_ids'], answers_tokenized['token_type_ids'],
                                      answers_tokenized['attention_mask'])), batch_size=batch_size)

        qoutputs = model(**questions_tokenized)
        questions_embeddings = mean_pooling(qoutputs[0], questions_tokenized['attention_mask'])
        questions_embeddings = questions_embeddings.cpu().detach().numpy()

        answers_embeddings = []
        for input_ids, token_type_ids, attention_mask in aloader:
            batch_a_tok = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            }
            aoutputs = model(**batch_a_tok)
            answers_embeddings.append(mean_pooling(aoutputs[0], batch_a_tok['attention_mask']).cpu().detach().numpy())
        answers_embeddings = np.concatenate(answers_embeddings, axis=0)

        d = answers_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(answers_embeddings)
        index.add(answers_embeddings)
        faiss.normalize_L2(questions_embeddings)

        recall_dict, ncdg_dict, map_dict = calculate_metrics_llm(questions_embeddings, index)

    metrics_dict = {
        "recall": recall_dict,
        "ncdg": ncdg_dict,
        "map": map_dict,
    }

    with open("data/metrics.json") as f:
        json.save(metrics_dict, f)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.mkdir("data")

    parser = argparse.ArgumentParser(
        description="QA experiments on BioASQ and RealMedQA")

    parser.add_argument(
        "--model-type",
        default="bert-base-uncased",
        type=str,
        help="Model to use for information retrieval.",
    )
    parser.add_argument(
        "--dataset-type",
        default="RealMedQA",
        type=str,
        help="Data to use for information retrieval: 'BioASQ' or 'RealMedQA'.",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Batch size used for inference'.",
    )
    args = parser.parse_args()
    main(args)
