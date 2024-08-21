[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CodeFactor](https://www.codefactor.io/repository/github/gck25/realmedqa/badge/main)](https://www.codefactor.io/repository/github/gck25/realmedqa/overview/main)

# RealMedQA
RealMedQA is a biomedical question answering dataset consisting of realistic question and answer pairs. The questions were created by medical students and a large language model (LLM), while the answers are guideline recommendations provided by the UK's National Institute for Health and Care Excellence (NICE).  This repositary contains the code to run experiments using the baseline models, i.e. Contriever, BM25, BERT, PubMedBERT, BioBERT, BioBERT fine-tuned on PubMedQA and SciBERT.  The full paper describing the dataset and the experiments has been accepted to the American Medical Informatics Association (AMIA) Annual Symposium and is available [here](https://arxiv.org/abs/2408.08624).

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

### Citation
If you use this codebase, please cite our work using the following reference:
```
@misc{kell2024realmedqapilotbiomedicalquestion,
      title={RealMedQA: A pilot biomedical question answering dataset containing realistic clinical questions}, 
      author={Gregory Kell and Angus Roberts and Serge Umansky and Yuti Khare and Najma Ahmed and Nikhil Patel and Chloe Simela and Jack Coumbe and Julian Rozario and Ryan-Rhys Griffiths and Iain J. Marshall},
      year={2024},
      eprint={2408.08624},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.08624}, 
}
```
