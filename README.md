# Mirror Question Pair

Problem formulation: Score question pairs on synonym criteria

WordEmbedding, CharEmbedding: word2vec, glove, decompose tfidf matrix using SVD (Singular Value Decomposition), decompose tfidf matrix using LDA (Latent Dirichlet Allocation), decompose tfidf matrix using NMF (Non-Negative Matrix Factorization). 

SentenceEmbedding: composed by Word/Char Embedding with weighting strategies - equally, or tf-idf linear, or tf-idf exponentially


Experiment 1: 
- Plenty hand-crafted features (e.g., Word Mover's Distance)
- LightGBM algorithm

Experiment 2:
- Take Siamese network architecture, apply different variations like CNN, RNN, RNN after CNN constructions as core layers.
The RNN part takes advantage of PyTorch pac_padded_sequence function to deal with variable length sentences. 

Experiment 3:
- Apply BIMPM(Bilateral Multi-Perspective Matching for Natural Language Sentences), in order to introduce corrections between two sub-networks of Siamese. 


Check full description and more details in https://sunnyyeti.github.io/2018/11/06/mirror-question-pair-detection/
