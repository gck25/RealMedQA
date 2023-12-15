# RealMedQA
RealMedQA is a biomedical question answering dataset consisting of realistic question and answer pairs. The questions were created by medical students and a large language model (LLM), while the answers are guideline recommendations provided by the UK's National Institute for Health and Care Excellence (NICE).  This repositary contains the code to run experiments using the baseline models, i.e. Contriever, BM25, BERT, PubMedBERT, BioBERT, BioBERT fine-tuned on PubMedQA and SciBERT.

## Requirements
Installing python environment
```
pip install -r requirements
```

## Run

### Example

```commandline
python main.py --model-type bm25 --dataset-type RealMedQA --batch-size 16 --seed 0
```

### Arguments
* `--model-type`: `str`
  * BM25: `bm25`
  * BERT: `bert-base-uncased`
  * SciBERT: `allenai/scibert_scivocab_uncased`
  * BioBERT: `dmis-lab/biobert-v1.1`
  * BioBERT fine-tuned on PubMedQA: `blizrys/biobert-v1.1-finetuned-pubmedqa`
  * PubMedBERT: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
  * Contriever: `facebook/contriever`
* `--dataset-type`: `str`
  * `RealMedQA`
  * `BioASQ`
* `--batch-size`: `int`
  * Batch size for encoding answers.
* `--seed`: `int`
  * Seed to initialize random sampler of BIoASQ QA pairs.

### Output
THe output is the JSON file `metrics.json` in the `data` directory with `nDCG@k` and `MAP@k` for
k $\in$ {2, 5, 10, 20, 50, 100} and `recall@k` for k $\in$ {1, 2, 5, 10, 20, 50, 100}.
