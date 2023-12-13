# RealMedQA
RealMedQA is a biomedical question answering dataset consisting of realistic question and answer pairs. The questions were created by medical students and a large language model (LLM), while the answers are guideline recommendations provided by the UK's National Institute for Health and Care Excellence (NICE).  This repo the code to run experiments using the baseline models, i.e. Contriever, BM25, BERT, PubMedBERT, BioBERT, BioBERT fine-tuned on PubMedQA and SciBERT.

## Requirements
Installing python environment
```
pip install -r requirements
```

## Dataset
To download and process the dataset, complete the following steps:

1. Create a `data` folder in this repositary.
2. Download the [BioASQ]([BeIR/bioasq-generated-queries](https://huggingface.co/datasets/BeIR/bioasq-generated-queries)https://huggingface.co/datasets/BeIR/bioasq-generated-queries) dataset.
3. Download the [RealMedQA](https://huggingface.co/datasets/k2141255/RealMedQA) dataset.
4. Move the BioASQ and RealMedQA datasets to the `data` folder.

### RealMedQA
