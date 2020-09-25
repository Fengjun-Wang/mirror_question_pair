# Mirror Question Pair

Problem formulation: Score question pairs on synonym criteria

This is an online competition hosted by one Chinese company which aims to predict whether a given pair of questions actually share the same meaning semantically. Because of data privacy protection, all original questions are encoded as sequences of char ID and word ID. Thus, we can't leverage on open source pre-trained sentence embeddings or word embeddings. The competition organers provide their own word embedding and char embedding for us. In this way, they keep the data privacy as well as fairness. 
Char may contain single Chinese word, single English letter, punctuation and space. Word may contain Chinese and English words, punctuation and space.

## Pre-processing
WordEmbedding, CharEmbedding: word2vec; glove; decompose tfidf matrix using SVD (Singular Value Decomposition), decompose tfidf matrix using LDA (Latent Dirichlet Allocation), decompose tfidf matrix using NMF (Non-Negative Matrix Factorization). 

SentenceEmbedding: get the sentence embedding directly - tfidf matrix embedding, decompose etc.;synthesized by Word/Char Embedding with weighting strategies - equally, or tf-idf linear, or tf-idf exponentially

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
Check full description and more details about model structures in https://sunnyyeti.github.io/post/mirror-question-pair-detection/

## Future dev
Well, after we develop the model, BERT becomes popular and is showing its super power in NLP domain. 
We can define own task processor (e.g., for our case, SimilarityProcessor), and apply bert with either fine-tuning or feature-based mode. 
