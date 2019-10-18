# GermEval 2019 Task 1 - Shared Task on Hierarchical Classification of Blurbs

This repository contains the code for the participation on GermEval 2019 Task 1 -- Shared task on hierarchical classification of blurbs.





## To run the experiments first install:

    pip install -r requirements.txt
    
    wget https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/de-wiki-fasttext-300d-1M.vectors.npy
    wget https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/de-wiki-fasttext-300d-1M
    
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords


# The following files run experiments for different strategies:

    subtask_a.py
    
    subtask_b_local_classifier.py
    
    subtask_b_global_classifier.py
    
    
# Evaluating an experiment:

    score.sh
    
    
# The competition data and evaluation scripts are still available through codalab:

- [https://competitions.codalab.org/competitions/20139](https://competitions.codalab.org/competitions/20139)
