import torch
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score


def get_question_answer_dict(answers, answers_unique):
    """
    Gets dictionary that maps from question indices to answer indices.

    Args:
        answers: list(str)
            original answers list
        answers_unique: list(str)
            list of unique answers

    Returns:
        q2a: dict(int): in
            dict that maps question indices to answer indices
    """
    q2a = {}
    for i, ans in enumerate(answers):
        ind = answers_unique.index(ans)
        q2a[i] = ind
    return q2a


def mean_pooling(token_embeddings, mask):
    """
    Applies mean pooling to token embeddings to create sentence embeddings.

    Args:
        token_embeddings: torch.tensor
            token embeddings created using LLM
        mask: torch.tensor
            mask to apply to embeddings

    Returns:
        sentence_embeddings: torch.tensor
            sentence embeddings for information retrieval
    """
    # apply mask to token embeddings
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)

    # calculate mean of token embeddings to create sentence embeddings
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def create_dataset_tokenized(dataset, tokenizer, device):
    """
    Tokenizes input dataset and allocates data to CPU or GPU.

    Args:
        dataset: list(str)
            question ar answer dataset
        tokenizer: AutoTokenizer
            LLM tokenizer to be applied to dataset
        device: torch.device
            CPU or GPU device

    Returns:
        dataset_tokenized: dict(str) = torch.tensor
            tokenized dataset allocated to device
    """
    # tokenize dataset
    dataset_tokenized = tokenizer(dataset, padding=True, truncation=True)

    # convert dataset to torch dataset and allocate to device
    dataset_tokenized = {
        'input_ids': torch.tensor(dataset_tokenized['input_ids']).to(device),
        'token_type_ids': torch.tensor(dataset_tokenized['token_type_ids']).to(device),
        'attention_mask': torch.tensor(dataset_tokenized['attention_mask']).to(device),
    }

    return dataset_tokenized


def calculate_metrics_llm(questions_embeddings, index, q2a):
    """
    Calculates recall@k, nDCG@k and MAP@k using LLM embeddings.

    Args:
        questions_embeddings: np.array
            LLM embeddings of questions
        index: faiss.IndexFlatIP
            faiss search index of answer embeddings
        q2a: dict(str) = str
            question to answer dict

    Returns:
        recall_dict, ndcg_dict, map_dict

        recall_dict: dict(str) = float
            recall@k for k in {1, 2, 5, 10, 20, 50, 100}
        ndcg_dict: dict(str) = float
            nDCG@k for k in {2, 5, 10, 20, 50, 100}
        map_dict: dict(str) = float
            MAP@k for k in {2, 5, 10, 20, 50, 100}
    """
    # get number of questions
    N = len(questions_embeddings)

    # initialize metrics dicts
    recall_dict = {}
    ndcg_dict = {}
    map_dict = {}

    # for each value of k
    for k in [1, 2, 5, 10, 20, 50, 100]:
        # initialize metrics for given value of k
        recall = 0
        ndcg_total = 0
        map_ = 0

        # for each question
        for i in range(N):
            # find indices and confidence scores of each answer
            A, I = index.search(questions_embeddings[i:i + 1], k)

            # trace unique answer index to original answer index
            ans_i = q2a[i]
            # if correct answer has been retrieved during search
            if ans_i in I:
                # increment recall@k
                recall += 1
                # for values of k greater than 1, calculate MAP@k and nDCG@k
                if k > 1:
                    # initialize each element of y_true to be 0 (False) or 1 (True) - binary labels required
                    y_true = np.array([0.0] * A)
                    ind = list(I[0]).index(ans_i)
                    y_true[0][ind] = 1.0

                    # calculate MAP@k and nDCG@k
                    ndcg_total += ndcg_score(y_true=y_true, y_score=A, k=k)
                    map_ += average_precision_score(y_true=y_true[0], y_score=A[0])

        # for values of k greater than 1, update MAP@k and nDCG@k dicts
        if k > 1:
            # normalize nDCG@k
            ndcg_total /= N
            # update nDCG@k dict
            ndcg_dict[f"ncdg@{k}"] = ndcg_total
            # normalize MAP@k
            map_ /= N
            # update MAP@k dict
            map_dict[f"map@{k}"] = map_

        # normalize recall@k
        recall /= N
        # update recall@k dict
        recall_dict[f"recall@{k}"] = recall

    return recall_dict, ndcg_dict, map_dict


def calculate_metrics_bm25(bm25, questions, tokenized_questions, answers, answers_unique):
    """
    Calculates recall@k, nDCG@k and MAP@k using BM25 scoring function.

    Args:
        bm25: rank_bm25.BM25Okapi
            BM25 object with tokenized answer.
        questions: list(str)
            list of untokenized questions
        tokenized_questions: list(list(str))
            list of tokenized questions
        answers: list(str)
            list of untokenized answers
        answers_unique:
            list of unique untokenized answers

    Returns:
        recall_dict, ndcg_dict, map_dict

        recall_dict: dict(str) = float
            recall@k for k in {1, 2, 5, 10, 20, 50, 100}
        ndcg_dict: dict(str) = float
            nDCG@k for k in {2, 5, 10, 20, 50, 100}
        map_dict: dict(str) = float
            MAP@k for k in {2, 5, 10, 20, 50, 100}
    """
    # get number of questions
    N = len(questions)

    # initialize metrics dicts
    recall_dict = {}
    ndcg_dict = {}
    map_dict = {}

    # initialize metrics for given value of k
    for k in [1, 2, 5, 10, 20, 50, 100]:
        # initialize metrics for given value of k
        recall = 0
        ndcg_total = 0
        map_ = 0

        # for each untokenized question, tokenized question and answer
        for q, tq, ans in zip(questions, tokenized_questions, answers):
            # get top k unique answers
            top_k = bm25.get_top_n(tq, answers_unique, n=k)
            # if answer is in top k list
            if ans in top_k:
                # increment recall@k
                recall += 1
                # if k is greate than 1, calculate nDCG@k and MAP@k
                if k > 1:
                    # find index of answer in original answer list
                    ind = top_k.index(ans)
                    # get BM25 scores for tokenized question
                    scores = bm25.get_scores(tq)

                    # calculate normalized BM25 score for each unique answer in top k list
                    y_score = [scores[i] for i, d in enumerate(answers_unique) if d in top_k]
                    norm = sum(y_score)
                    y_score = np.array([score / norm for score in y_score]).reshape(1, -1)

                    # initialize each element of y_true to be 0 (False) or 1 (True) - binary labels required
                    y_true = np.array([0] * k).reshape(1, -1)
                    y_true[0][ind] = 1

                    # calculate MAP@k and nDCG@k
                    ndcg_total += ndcg_score(y_true=y_true, y_score=y_score, k=k)
                    map_ += average_precision_score(y_true=y_true[0], y_score=y_score[0], pos_label=1)

        # for values of k greater than 1, update MAP@k and nDCG@k dicts
        if k > 1:
            # normalize nDCG@k
            ndcg_total /= N
            # update nDCG@k dict
            ndcg_dict[f"ncdg@{k}"] = ndcg_total
            # normalize MAP@k
            map_ /= N
            # update MAP@k dict
            map_dict[f"map@{k}"] = map_

        # normalize recall@k
        recall /= N
        # update recall@k dict
        recall_dict[f"recall@{k}"] = recall

    return recall_dict, ndcg_dict, map_dict
