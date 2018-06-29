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
        - seq2seq 
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

[bacth lstm pytorch_1](https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e)

[bacth lstm pytorch_2](https://github.com/ngarneau/understanding-pytorch-batching-lstm/blob/master/Understanding%20Pytorch%20Batching.ipynb)


[Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
[paper siamese Manhattan LSTM](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
https://zhuanlan.zhihu.com/p/26996025


[Text CNN2](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf)
[Text CNN1](http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/)

[Semantic matching in seach 李航](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/ml_for_match-step2.pdf)

[Sementic similarity Quora deep learning methods](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)

https://web.stanford.edu/class/cs224n/reports/2759336.pdf

https://arxiv.org/pdf/1702.03814.pdf

https://arxiv.org/pdf/1702.03814.pdf

https://arxiv.org/pdf/1606.01933.pdf

[set forget bias 1](https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/3)

[Attention-Based Convolutional Neural Network
for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf)
## TO DO
* [LDA NNE Embedding](http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py): 
    - ~~all kinds of distance~~
    - ~~absdiff~~
* Add original embedding:
    - ~~Word SVD absdiff (word tfidf SVD 分解得到的sentence embed)~~
    - ~~Char SVD absdiff (char tfidf SVD 分解得到的sentence embed)~~
    - ~~Glove tfidf linear word (becasue word2vec already used equally)~~
    - ~~Glove tfidf linear char~~
    - ~~not add SVD word/char to sentence~~
    - maybe remove SVD word/char to sentence 
* Add meta feature:
    - ~~question ends the same, or starts the same~~
    
* Add graph feature:
    - page rank

* Based on LSTM v4:
    - ~~try weight initialization~~
    - try big hidden size(lstm and fcn)
    - ~~is_freeze False~~
    - ~~augmentation~~
    - try chars
    - try CNN plus LSTM


* CNN+LSTM 不是很好
* Similarity in CNN
* Chars in CNN
* Seq2Seq
* try clustering in the feature space with high importance from lgm as new feature, maybe serve for stacking
 
 * Stacking; multi-layer stacking; do cv when stacking; choose models near 0.2
 * 所以争取多做一些模型到0.2以内，最好是不同的结构，现在只是在lstm上面,争取CNN,seq2seq也到0.2以内
 * 要不要神经网络中加入hand-crafted feature
 * CNN 的dropout 可以用在pooling 之后，用在之前是不是作用不是很明显，因为dropout的不是最大值，就没啥用
 * lstm 不只用最后的state,或者不是简单的拼接,比如加上减法，相乘
 * lstm 加上interaction
 * CNN 
    - atttention
    - similarity non-similarity 这篇论文