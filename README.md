# Mirror Question Pair

Problem formulation: Score question pairs on synonym criteria

## Pre-processing
WordEmbedding, CharEmbedding: word2vec, glove, decompose tfidf matrix using SVD (Singular Value Decomposition), decompose tfidf matrix using LDA (Latent Dirichlet Allocation), decompose tfidf matrix using NMF (Non-Negative Matrix Factorization). 

SentenceEmbedding: composed by Word/Char Embedding with weighting strategies - equally, or tf-idf linear, or tf-idf exponentially

## Modeling

Experiment 1: 
- Plenty hand-crafted features (e.g., Word Mover's Distance)
- LightGBM algorithm

Experiment 2:
- Take Siamese network architecture, apply different variations like CNN, RNN, RNN after CNN constructions as core layers.
The RNN part takes advantage of PyTorch pac_padded_sequence function to deal with variable length sentences. 

Experiment 3:
- Apply BIMPM(Bilateral Multi-Perspective Matching for Natural Language Sentences), in order to introduce corrections between two sub-networks of Siamese. BIMPM considers relationships among chars and also among words, while introduce more parameters to tune at the same time. 

(Diff versioning folders are for convenience of recover/ensemble during the fast-paced competition period. )

## Full description of this project
Check full description and more details in https://sunnyyeti.github.io/2018/11/06/mirror-question-pair-detection/

## Future dev
Well, after we develop the model, BERT becomes popular and is showing its super power in NLP domain. 
We can define own task processor (e.g., for our case, SimilarityProcessor), and apply bert with either fine-tuning or feature-based mode. 
