## Ideas
* 传统机器学习方法
    - tf-idf 
        - 两个句子的差
        - 两个句子整体的
        - word space and letter space
    - tf_idf的svd分解
        - 句子单独分解求差
        - 两个句子整体的分解
        - word space and letter space
    - sentence_embedding
        - find paper
        - simply (weighted)average of word2vec embedding
            - weight could be done according to tf-idf value 
        - [WMD](https://towardsdatascience.com/sentence-embedding-3053db22ea77)
    - get other embedding 
        - glove
        - fasttex
* neural network
    - Siamese
        - CNN
        - RNN
* vector distance
    - cosine similarity
    - L2
    - L1 
    - Sigmoid kerne
    - RBF kernel
    - Laplacian kernel
    - Chi-squared kernel
## Resources:
[Quora question pair 1st solution](https://www.kaggle.com/c/quora-question-pairs/discussion/34355)

[MatchZoo](https://github.com/faneshion/MatchZoo)

    
https://github.com/YuriyGuts/kaggle-quora-question-pairs

## TO DO
* [LDA NNE Embedding](http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py): 
    - all kinds of distance
    - absdiff
* Add original embedding:
    - Word SVD absdiff
    - Char SVD absdiff
    - Glove tfidf linear word (becasue word2vec already used equally)
    - Glove tfidf linear char
    - not add SVD word/char to sentence
    - maybe remove SVD word/char to sentence 
* Add meta feature:
    - question ends the same, or starts the same
    
* Add graph feature:
    - page rank
