import os
import json
import faiss
import torch
import random
import argparse
import gensim
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from utils import (mean_pooling, create_dataset_tokenized, calculate_metrics_llm, calculate_metrics_bm25,
                   get_question_answer_dict)


def main(args):
    model_type = args.model_type
    dataset_type = args.dataset_type
    batch_size = args.batch_size
    seed = args.seed

    # decide which dataset to use: BioASQ or RealMedQA
    if dataset_type == "BioASQ":
        # load BioASQ retrieval dataset used by BEIR
        dataset = load_dataset("BeIR/bioasq-generated-queries", split="train")

        # obtain list of possible BioASQ indices
        poss_inds = range(len(dataset))

        # set random seed
        random.seed(seed)

        # randomly sample 230 indices
        inds = random.sample(poss_inds, 230)

        # create test dataset using sampled indices
        dataset_test = dataset[inds]

        # select questions from indices
        questions = dataset_test['query']

        # create answers using titles and abstracts of BioASQ papers
        answers = [f'{t} {a}' for t, a in zip(dataset_test['title'], dataset_test['text'])]

        # strip questions and answers of leading and tailing spaces
        questions = [q.strip() for q in questions]
        answers = [a.strip() for a in answers]
    else:
        # load RealMedQA dataset
        dataset = load_dataset("k2141255/RealMedQA", split="train")

        # get QA pairs that are 'completely' plausible and answered
        filtered_dataset = dataset.filter(lambda example: (example["Plausible"] == 'Completely') &
                                                          (example['Answered'] == 'Completely'))

        # strip questions and answers of trailing and leading spaces
        questions = [q.strip() for q in filtered_dataset['Question']]
        answers = [a.strip() for a in filtered_dataset['Recommendation']]

    # get list of unique answers
    answers_unique = list(set(answers))

    # decide which model to use: BM25 or BERT-based LLM
    if model_type == "bm25":
        # process and tokenize questions and answers
        tokenized_answers = [list(gensim.utils.tokenize(doc.lower())) for doc in answers_unique]
        tokenized_questions = [list(gensim.utils.tokenize(q.lower())) for q in questions]

        # initialize BM25
        bm25 = BM25Okapi(tokenized_answers)

        # calculate recall@k, nDCG@k and MAP@k for different values of k
        recall_dict, ndcg_dict, map_dict = calculate_metrics_bm25(bm25, questions, tokenized_questions, answers,
                                                                  answers_unique)
    else:
        # if available use GPU, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load relevant model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModel.from_pretrained(model_type).to(device)

        # tokenize questions and answers
        questions_tokenized = create_dataset_tokenized(questions, tokenizer, device)
        answers_tokenized = create_dataset_tokenized(answers_unique, tokenizer, device)

        # create question embeddings
        qoutputs = model(**questions_tokenized)
        questions_embeddings = mean_pooling(qoutputs[0], questions_tokenized['attention_mask'])
        questions_embeddings = questions_embeddings.cpu().detach().numpy()

        # create answer loader
        aloader = DataLoader(list(zip(answers_tokenized['input_ids'], answers_tokenized['token_type_ids'],
                                      answers_tokenized['attention_mask'])), batch_size=batch_size)

        # create answer embeddings
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

        # create faiss search index of answer embeddings
        d = answers_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)

        # normalize questions and answers with L2 norm to search with dot product (using inner product operation)
        faiss.normalize_L2(answers_embeddings)
        index.add(answers_embeddings)
        faiss.normalize_L2(questions_embeddings)

        q2a = get_question_answer_dict(answers, answers_unique)

        # calculate recall@k, nDCG@k and MAP@k using different values of k
        recall_dict, ndcg_dict, map_dict = calculate_metrics_llm(questions_embeddings, index, q2a)

    metrics_dict = {
        "recall": recall_dict,
        "ncdg": ndcg_dict,
        "map": map_dict,
    }

    with open("data/metrics.json", 'w') as f:
        json.dump(metrics_dict, f)


if __name__ == "__main__":
    # create data path to store results of experiments
    if not os.path.exists("data"):
        os.mkdir("data")

    parser = argparse.ArgumentParser(
        description="QA experiments on BioASQ and RealMedQA")

    parser.add_argument(
        "--model-type",
        default="bm25",
        type=str,
        help="Model to use for information retrieval.",
    )
    parser.add_argument(
        "--dataset-type",
        default="BioASQ",
        type=str,
        help="Data to use for information retrieval: 'BioASQ' or 'RealMedQA'.",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Batch size used for inference.",
    )
    parser.add_argument(
        "--seed",
        default=8,
        type=int,
        help="Random seed used for random sampler of BioASQ.",
    )
    args = parser.parse_args()
    main(args)
