# mirror_question_pair
Score question pairs on synonym criteria

Check full description in https://github.com/sunnyyeti/MaterialsPreparation/blob/master/projects.md , copy for reading:  

# Projects
## Mirror question pair classification
### Problem statement
This is an online competition hosted by one Chinese company which aims to predict whether a given pair of questions  actually share the same meaning semantically. Because of privacy, all original questions are encoded as sequences of char ID and word ID. Char may contain single Chinese word, single English letter, punctuation and space. Word may contain Chinese and English words, punctuation and space. 

### Dataset desciption
* char_embed.txt  
300-dimension char embedding vector trained by Google word2vec.
* word_embed.txt  
300-dimension word embedding vector trained by Google word2vec.
* question.csv  
Contains all questions in the train set and test set. Each question contains sequence of char ID and word ID.
* train.csv  
Question pairs in the train set.
* test.csv  
Question pairs in the test set.

### Solution
I tried both traditional methods and end-to-end deep learning models. Results show that deep learning models outperform traditional ones that in addition need a lot of hand-crafted features.
#### Traditional method
##### Feature engineering
The main idea is that we need to get the embedding vector of the sentence. There are mainly two ways, one is to synthesize the sentence embedding from word/char embeddings, and another one is to get the sentence embedding directly.

Get the sentence embedding directly:
* words_3gram_sentence_tfidf_embedding
* chars_5gram_sentence_tfidf_embedding
* words_3gram_tfidf_SVD_sentence_embed
* chars_5gram_tfidf_SVD_sentence_embed
* words_1gram_tfidf_NMF_sentence_embed (Non-Negative Matrix Factorization)
* chars_1gram_tfidf_NMF_sentence_embed
* words_1gram_tf_LDA_sentence_embed (LatentDirichletAllocation)
* chars_1gram_tf_LDA_sentence_embed

Synthesize the sentence embedding from word/char embedding.

Methods to get the word/char embedding:
1. word2vec word/char embedding supplied by the host
2. glove word/char embedding trained by ourselves
3. decompose the tf-idf matrix using SVD

We tried three different weighting strategies. 
1. equally
2. tf-idf linear weight
3. tf-idf exp weight

So all sentence embeddings we get by synthesis:
* word2vec_word_embed_to_sentence_embed_mode_equally
* word2vec_word_embed_to_sentence_embed_mode_tf_idf_exp
* word2vec_word_embed_to_sentence_embed_mode_tf_idf_linear
* word2vec_char_embed_to_sentence_embed_mode_equally
* word2vec_char_embed_to_sentence_embed_mode_tf_idf_exp
* word2vec_char_embed_to_sentence_embed_mode_tf_idf_linear
* glove_char_embed_to_sentence_embed_mode_equally
* glove_char_embed_to_sentence_embed_mode_tf_idf_exp
* glove_char_embed_to_sentence_embed_mode_tf_idf_linear
* glove_word_embed_to_sentence_embed_mode_equally
* glove_word_embed_to_sentence_embed_mode_tf_idf_exp
* glove_word_embed_to_sentence_embed_mode_tf_idf_linear
* words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_equally
* words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_tf_idf_exp
* words_3gram_tfidf_SVD_word_embed_to_sentence_embed_mode_tf_idf_linear
* chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_equally
* chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_tf_idf_exp
* chars_5gram_tfidf_SVD_char_embed_to_sentence_embed_mode_tf_idf_linear

After getting the sentence embeddings, we can form features by manipulating embedding pairs. One is calculating some distance between these two embeddings, the other one is keeping the absolute difference of these two embedding vectors as features.

The distance we tried for tf-idf sentence embedding:
* cosine_similarity
  $$
  cos(x,y) = \frac{x^Ty}{||x||_2||y||_2}
  $$

* polynomial_kernel  
The __*polynomial_kernel*__ computes the degree-$d$ polynomial kernel between two vectors. The polynomial kernel represents the similarity between two vectors. Conceptually, the polynomial kernels considers not only the similarity between vectors under the same dimension, but also across dimensions. When used in machine learning algorithms, this allows to account for feature interaction.
The polynomial kernel is defined as:
$K(x,y) = (\gamma x^Ty+c_0)^d$, where $\gamma$ defaults to $\frac{1}{|x|}$, where $|x|$ means the number of elements in $x$; $c_0$ defaults to $1$; $d$ defaults to 3.

* sigmoid_kernel  
The function **_sigmoid_kernel_** computes the sigmoid kernel between two vectors. The sigmoid kernel is also known as hyperbolic tangent, or Multilayer Perceptron (because, in the neural network field, it is often used as neuron activation function). It is defined as: $K(x,y)=tanh(\gamma x^Ty+c_0)$, where $\gamma$ defaults to $\frac{1}{|x|}$, where $|x|$ means the number of elements in $x$; $c_0$ defaults to $1$; 

* rbf_kernel  
The function **_rbf_kernel_** computes the radial basis function (RBF) kernel between two vectors. This kernel is defined as: $K(x,y) = exp(-\gamma ||x-y||_2^2)$, where $\gamma$ defaults to $\frac{1}{|x|}$, where $|x|$ means the number of elements in $x$.
* laplacian_kernel  
The function **_laplacian_kernel_** is a variant on the radial basis function kernel defined as:$K(x,y) = exp(-\gamma ||x-y||_1)$, where $||x-y||_1$ is the Manhattan distance between the input vector and  $\gamma$ defaults to $\frac{1}{|x|}$, where $|x|$ means the number of elements in $x$.

* my_chi2_kernel  
The **_chi squared kernel_** is given by $K(x,y)=exp(-\gamma \sum_{i}\frac{(x[i]-y[i])^2}{x[i]+y[i]})$, where $\gamma$ defaults to $1$. The data is assumed to be non-negative, so this kernel is only applied to tf-idf sentence embedding.
* euclidean  
$K(x,y)=||x-y||_2$
* cityblock  
$K(x,y)=||x-y||_1$
* WMD (Word Mover's Distance)  
Assume sentence 1 is represented as bag of words $b_1=\{w_1,w_2,\dots, w_n\}$ and sentence 2 is represented as bag of words $b_2=\{\bar{w}_1,\bar{w}_2,\dots, \bar{w}_m\}$. Then it is defined as $K(b1,b2)=\sum_{i=1}^{n}\min_{j}D(w_i,\bar{w}_j)$, and $D$ is a function calculating distance between two word embeddings. It could be euclidean distance normally.
  
The distance we tried for non-tf-idf sentence embedding:
* cosine_similarity,
* linear_kernel  
Linear kernel is defined as $K(x,y)=x^Ty$
* polynomial_kernel
* sigmoid_kernel
* rbf_kernel
* laplacian_kernel
* my_chi2_kernel
* euclidean
* cityblock

The distance we tried for tf-idf sentence embedding(element is non-negative):
* cosine_similarity,
* polynomial_kernel
* sigmoid_kernel
* rbf_kernel
* laplacian_kernel
* euclidean
* cityblock

The word embeeding we adopted to calculate the WMD distance:
* words_3gram_tfidf_SVD_word_embed
* word2vec_word_embed
* glove_word_embed.csv
* chars_5gram_tfidf_SVD_char_embed.csv
* word2vec_char_embed.csv
* glove_char_embed.csv


We also calculate the absolute element-wise difference of two embeddings as features, and these features keep the original information in the embeddings. The embeddings used:
* words_3gram_tfidf_SVD_sentence_embed
* chars_5gram_tfidf_SVD_sentence_embed
* glove_char_embed_to_sentence_embed_mode_tf_idf_linear
* glove_word_embed_to_sentence_embed_mode_tf_idf_linear
* words_1gram_tf_LDA_sentence_embed
* chars_1gram_tf_LDA_sentence_embed
* chars_1gram_tfidf_NMF_sentence_embed
* words_1gram_tfidf_NMF_sentence_embed
                       

Also we calculate some meta features like:
* F.spair_len_s1
* F.spair_len_s2
* F.spair_len_dif_abs
* F.spair_len_dif_over_max
* F.spair_len_dif_over_min
* F.spair_len_dif_over_mean
* F.levenshtein
* F.if_starts_same
* F.if_ends_same
* F.num_commom_n_gram(i) for i in range(1,n+1)
* F.jaccard_till_n_gram(n)

These are all features we used in traditional model. And the model we tried is LightGBM.


#### Deep learning method

We adopt the so-called siamese structure as our deep learning model.
Generally speaking, siamese network consists of a pair of sub networks who share the same paramters. For a pair of questions, we pass one question through one sub net and second question through another sub net, then the outputs of the two networks are concatnated together and put in a fully connected layer. Finally the output of the sigmoid gives us the probability of that the two questions have the same meaning.

##### CNN method
The sub network is CNN. 1D CNN for text looks like:
<div style="text-align:center"  ><img src ="./pics/text_CNN.png"  width=780px/></div>. Max pooling along the length dimension makes sure that questions with different lengths have outputs of same length. Some key codes are attached here:

```python
def forward(self, input):
    q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
    q1_embed = self.nn_Embedding(q1).transpose(1,2) ##NxLxC -> NxCxL
    q2_embed = self.nn_Embedding(q2).transpose(1,2)

    q1_conv1 = F.relu(self.conv1d_size2(q1_embed)) ##NxCxL
    q1_pool1,_ = q1_conv1.max(dim=2) ##NxC
    q1_conv2 = F.relu(self.conv1d_size3(q1_embed)) ##NxCxL
    q1_pool2,_ = q1_conv2.max(dim=2) ##NxC
    q1_conv3 = F.relu(self.conv1d_size4(q1_embed)) ##NxCxL
    q1_pool3,_ = q1_conv3.max(dim=2) ##NxC(100)
    q1_concat = torch.cat((q1_pool1,q1_pool2,q1_pool3),dim=1) ## Nx(c1+c2...)[300]

    q2_conv1 = F.relu(self.conv1d_size2(q2_embed)) ##NxCxL
    q2_pool1,_ = q2_conv1.max(dim=2) ##NxC
    q2_conv2 = F.relu(self.conv1d_size3(q2_embed)) ##NxCxL
    q2_pool2,_ = q2_conv2.max(dim=2) ##NxC
    q2_conv3 = F.relu(self.conv1d_size4(q2_embed)) ##NxCxL
    q2_pool3,_ = q2_conv3.max(dim=2) ##NxC(100)
    q2_concat = torch.cat((q2_pool1,q2_pool2,q2_pool3),dim=1) ## Nx(c1+c2...)[300]

    q_concat = torch.cat((q1_concat,q2_concat),dim=1) ##Nx600
    h1 = F.relu(self.out_hidden1(q_concat))
    h2 = F.relu(self.out_hidden2(h1))
    outscore = self.out_put(h2)
    return outscore
```

##### RNN method
The subnetwork is two-layer bidirectional LSTM. A one-layer bidirectional RNN looks like:
<div style="text-align:center"  ><img src ="./pics/bi_rnn.png"  width=780px/></div>

I utilize the `pack_padded_sequence` function from Pytorch to handle variable length inputs. Some key codes are attahced here:
```python
    def sort(self, input_tensor):
        input_lengths = torch.LongTensor(
            [torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in xrange(input_tensor.size(0))])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        _, reverse_perm_idx = perm_idx.sort(0)
        input_seqs = input_tensor[perm_idx][:, :input_lengths.max()]
        return input_seqs, input_lengths, reverse_perm_idx

    def forward(self, input):
        q1, q2 = torch.chunk(input, 2, dim=1)  ## Split the question pairs
        q1, q1_lens, q1_reverse_order_indx = self.sort(q1)
        q2, q2_lens, q2_reverse_order_indx = self.sort(q2)
        q1_pad_embed = self.nn_Embedding(q1)  ##NxLxC
        q2_pad_embed = self.nn_Embedding(q2)  ##NxLxC
        q1_embed = self.input_dropout(q1_pad_embed)
        q2_embed = self.input_dropout(q2_pad_embed)
        q1_pack_pad_seq_embed = pack_padded_sequence(q1_embed, batch_first=True, lengths=q1_lens)
        q2_pack_pad_seq_embed = pack_padded_sequence(q2_embed, batch_first=True, lengths=q2_lens)

        q1_out, q1_hidden = self.lstm(q1_pack_pad_seq_embed)
        q1h, q1c = q1_hidden

        q2_out, q2_hidden = self.lstm(q2_pack_pad_seq_embed)
        q2h, q2c = q2_hidden

        if self.bidirectional:
            q1_encode = torch.cat((q1h[-2], q1h[-1]), dim=1)
            q2_encode = torch.cat((q2h[-2], q2h[-1]), dim=1)
        else:
            q1_encode = q1h[-1]
            q2_encode = q2h[-1]
        q1_encode_reverse = q1_encode[q1_reverse_order_indx]
        q2_encode_reverse = q2_encode[q2_reverse_order_indx]

        q_pair_encode_q12 = torch.cat((q1_encode_reverse, q2_encode_reverse), dim=1)  ##TODO augment q1,q2 ; q2,q1
        q_pair_encode_q21 = torch.cat((q2_encode_reverse, q1_encode_reverse), dim=1)
        q_pair_encode = torch.cat((q_pair_encode_q12, q_pair_encode_q21), dim=0)
        h1 = self.linear1_dropout(F.relu(self.linear1(q_pair_encode)))
        out = self.linear2(h1)
        out1, out2 = torch.chunk(out, 2, dim=0)
        return (out1 + out2) / 2
```

##### CNN+RNN method
Add a CNN layer before RNN layer. The CNN layer is "same padding". Some key codes are  attached here:
```python
    def sort(self,input_tensor):
        input_lengths = torch.LongTensor([torch.max(input_tensor[i, :].data.nonzero()) + 1 for i in xrange(input_tensor.size(0))])
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        _,reverse_perm_idx = perm_idx.sort(0)
        input_seqs = input_tensor[perm_idx][:, :input_lengths.max()]
        return input_seqs,input_lengths,reverse_perm_idx

    def forward(self, input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1,q1_lens,q1_reverse_order_indx = self.sort(q1)
        q2,q2_lens,q2_reverse_order_indx = self.sort(q2)
        q1_pad_embed = self.nn_Embedding(q1).transpose(1,2) ##NxLxC->NxCxL
        q2_pad_embed = self.nn_Embedding(q2).transpose(1,2) ##NxLxC->NxCxL
        q1_conv_out = F.relu(self.batch_norm(self.input_conv(q1_pad_embed))).transpose(1,2)
        q2_conv_out = F.relu(self.batch_norm(self.input_conv(q2_pad_embed))).transpose(1,2)
        q1_embed = self.input_dropout(q1_conv_out)
        q2_embed = self.input_dropout(q2_conv_out)
        q1_pack_pad_seq_embed = pack_padded_sequence(q1_embed, batch_first=True, lengths=q1_lens)
        q2_pack_pad_seq_embed = pack_padded_sequence(q2_embed, batch_first=True, lengths=q2_lens)

        q1_out,q1_hidden = self.lstm(q1_pack_pad_seq_embed)
        q1h,q1c = q1_hidden

        q2_out,q2_hidden = self.lstm(q2_pack_pad_seq_embed)
        q2h,q2c = q2_hidden

        if self.bidirectional:
            q1_encode = torch.cat((q1h[-2],q1h[-1]),dim=1)
            q2_encode = torch.cat((q2h[-2],q2h[-1]),dim=1)
        else:
            q1_encode = q1h[-1]
            q2_encode = q2h[-1]
        q1_encode_reverse = q1_encode[q1_reverse_order_indx]
        q2_encode_reverse = q2_encode[q2_reverse_order_indx]

        q_pair_encode_q12= torch.cat((q1_encode_reverse,q2_encode_reverse),dim=1)##TODO augment q1,q2 ; q2,q1
        q_pair_encode_q21 = torch.cat((q2_encode_reverse,q1_encode_reverse),dim=1)
        q_pair_encode = torch.cat((q_pair_encode_q12,q_pair_encode_q21),dim=0)
        h1 = self.linear1_dropout(F.relu(self.linear1(q_pair_encode)))
        out = self.linear2(h1)
        out1,out2 = torch.chunk(out,2,dim=0)
        return (out1+out2)/2
```

##### BIMPM 
We do not include any correlation between two sub networks in previous versions. BIMPM introduces the correlation between two networks. The reference paper: [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf)
The general architecture looks like:
<div style="text-align:center"  ><img src ="./pics/BIMPM.png"  width=780px/></div>

The key part is the matching layer. The goal of this layer is to compare each contextual embedding (time-step) of one sentence against all contextual embeddings (time-steps) of the other sentence. First, we define a multi-perspective consine matching function $f_m$ to compare two vectors:
$$m=f_m(v_1,v_2;W)$$
where $v_1$ and $v_2$ are two $d$-dimensional vectors, $W \in R^{l*d}$ is a trainable parameter with the shape $l*d$, $l$ is the number of perspective and the returned value $m$ is $l$-dimensional vector, where $m_k=cosine(v_1 \odot W_k,v_2\odot W_k)$. $\odot$ is element-wise multiplication and $W_k$ is the $k_{th}$ row of W. Four kind of matchings are defined here:
<div style="text-align:center"  ><img src ="./pics/matching_4types.png"  width=780px/></div>

* Full matching 
  $$\overrightarrow{m_i}^{full} = f_m(\overrightarrow{h_i}^p,\overrightarrow{h_N}^q;W^1)$$
   $$\overleftarrow{m_i}^{full} = f_m(\overleftarrow{h_i}^p,\overleftarrow{h_1}^q;W^2)$$

* Maxpooling-Matching
 $$\overrightarrow{m_i}^{max} = max_{j\in(1...N)}f_m(\overrightarrow{h_i}^p,\overrightarrow{h_j}^q;W^3)$$
  $$\overleftarrow{m_i}^{max} = max_{j\in(1...N)}f_m(\overleftarrow{h_i}^p,\overleftarrow{h_j}^q;W^4)$$

* Attentive-Matching
$$\overrightarrow{a_{i,j}} = cosine(\overrightarrow{h_i}^p,\overrightarrow{h_j}^q) \ \ \ \ \ \ \ \ \ \ j=1,2,3...N$$
$$\overleftarrow{a_{i,j}} = cosine(\overleftarrow{h_i}^p,\overleftarrow{h_j}^q) \ \ \ \ \ \ \ \ \ \ j=1,2,3...N$$
$$\overrightarrow{h_i}^{mean} = \frac{\sum_j^N \overrightarrow{a_{i,j}}*\overrightarrow{h_j}^q}{\sum_j^N \overrightarrow{a_{i,j}}}$$
$$\overleftarrow{h_i}^{mean} = \frac{\sum_j^N \overleftarrow{a_{i,j}}*\overleftarrow{h_j}^q}{\sum_j^N \overleftarrow{a_{i,j}}}$$

 $$\overrightarrow{m_i}^{att} = f_m(\overrightarrow{h_i}^p,\overrightarrow{h_i}^{mean};W^5)$$
 $$\overleftarrow{m_i}^{att} = f_m(\overleftarrow{h_i}^p,\overleftarrow{h_i}^{mean};W^6)$$

* Max-Attentive-Matching
  
    This strategy is similar to the attentive-Matching strategy. However, instead of taking the weighed sum of all the contextual embeddings as the attentive
vector, we pick the contextual embedding with the highest cosine similarity as the attentive vector.

We apply all these four matching strategies to each timestep of the sentence P, and concatenate the generated eight
vectors as the matching vector for each time-step of P. We
also perform the same process for the reverse matching direction.
Some key codes of BIMPM are attached here:
```python
    def full_matching(self,q1_NLC,q2_NC,W):
        """
        :param q1_NLC:
        :param q2_NC:
        :return:NLP(p is number of perspective)
        """
        q1_NL1C = q1_NLC.unsqueeze(2)
        W_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NL1C*W_11PC
        q2_N11C = q2_NC.unsqueeze(1).unsqueeze(1)
        q2_N1PC = q2_N11C*W_11PC
        q1_mm_q2_NLPC = q1_NLPC*q2_N1PC
        q1_mm_q2_NLP = q1_mm_q2_NLPC.sum(dim=3)
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)
        q2_N1PC_norm = q2_N1PC.norm(p=2,dim=3)
        return q1_mm_q2_NLP/((q1_NLPC_norm*q2_N1PC_norm).clamp(min=1e-8))


    def maxpool_matching(self,q1_NLC,q2_NLC,W,q2_lengths):
        compare_L = torch.zeros(q2_NLC.size(1),q1_NLC.size(0),q1_NLC.size(1),self.number_perspective).to(self.device)
        for l in xrange(q2_NLC.size(1)):
            tmp_h_NC = q2_NLC[:,l,:]
            compare_L[l] = self.full_matching(q1_NLC,tmp_h_NC,W)
        res = torch.zeros(q1_NLC.size(0),q1_NLC.size(1),self.number_perspective).to(self.device)
        for i,l in enumerate(q2_lengths):
            res[i] = compare_L[:l,i].max(dim=0)[0]
        return res


    def attentive_matching(self,q1_NLC,q2_NLC,W):
        q1_NLC_norm = q1_NLC/(q1_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NLC_norm = q2_NLC/(q2_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NCL_norm = q2_NLC_norm.transpose(1,2)
        q1_q2_NLL = q1_NLC_norm.bmm(q2_NCL_norm)
        q1_q2_NLL_norm = q1_q2_NLL/q1_q2_NLL.sum(dim=2,keepdim=True).clamp(min=1e-8)
        q1_L_mean_NLC = q1_q2_NLL_norm.bmm(q2_NLC)

        q1_NLC_exp_NL1C = q1_NLC.unsqueeze(2)
        W_exp_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NLC_exp_NL1C * W_exp_11PC

        q1_L_mean_NLPC = q1_L_mean_NLC.unsqueeze(2)* W_exp_11PC ##NL1C * 11PC

        q1_L_mean_NLP_sum = (q1_NLPC*q1_L_mean_NLPC).sum(dim=3)##NLP
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)##NLP
        q1_L_mean_NLPC_norm = q1_L_mean_NLPC.norm(p=2,dim=3)##NLP

        return q1_L_mean_NLP_sum/((q1_NLPC_norm*q1_L_mean_NLPC_norm).clamp(min=1e-8))

    def max_attentive_matching(self,q1_NLC,q2_NLC,W,q2_lengths):
        q1_NLC_norm = q1_NLC/(q1_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NLC_norm = q2_NLC/(q2_NLC.norm(p=2,dim=2,keepdim=True).clamp(min=1e-8))
        q2_NCL_norm = q2_NLC_norm.transpose(1,2)
        q1_q2_NLL_cosine = q1_NLC_norm.bmm(q2_NCL_norm)
        res = torch.zeros(q1_NLC.size(0),q1_NLC.size(1),q1_NLC.size(2)).to(self.device)##NLC
        for i,l in enumerate(q2_lengths):
            tmp = q1_q2_NLL_cosine[i,:,:l]
            inds = tmp.max(dim=1)[1]
            res[i] = q2_NLC[i,inds,:]

        q1_NLC_exp_NL1C = q1_NLC.unsqueeze(2)
        W_exp_11PC = W.unsqueeze(0).unsqueeze(0)
        q1_NLPC = q1_NLC_exp_NL1C * W_exp_11PC

        q1_max_NLPC = res.unsqueeze(2)* W_exp_11PC ##NL1C * 11PC

        q1_max_NLP_sum = (q1_NLPC*q1_max_NLPC).sum(dim=3)##NLP
        q1_NLPC_norm = q1_NLPC.norm(p=2,dim=3)##NLP
        q1_max_NLPC_norm = q1_max_NLPC.norm(p=2,dim=3)##NLP

        return q1_max_NLP_sum/((q1_NLPC_norm*q1_max_NLPC_norm).clamp(min=1e-8))


    def forward(self,input):
        q1,q2  = torch.chunk(input, 2, dim=1) ## Split the question pairs
        q1, q1_lens, q1_perm_idx, q1_reverse_order_indx = self.sort(q1)
        q2, q2_lens, q2_perm_idx, q2_reverse_order_indx = self.sort(q2)
        q1_pad_embed = self.nn_Embedding(q1)##NLC
        q2_pad_embed = self.nn_Embedding(q2)##NLC
        q1_embed = self.input_drop(q1_pad_embed)
        q2_embed = self.input_drop(q2_pad_embed)
        q1_pack_pad_seq_embed = pack_padded_sequence(q1_embed, batch_first=True, lengths=q1_lens)
        q2_pack_pad_seq_embed = pack_padded_sequence(q2_embed, batch_first=True, lengths=q2_lens)
        ##Q1
        q1_out,q1_hidden = self.crl(q1_pack_pad_seq_embed)
        pad_q1_out = pad_packed_sequence(q1_out, batch_first=True)
        q1h,_ = q1_hidden
        pad_q1_forward,pad_q1_back = torch.chunk(pad_q1_out[0],2,dim=2)##NLC
        h_q1_forward = q1h[-2]
        h_q1_back = q1h[-1]
        pad_q1_forward_orig = pad_q1_forward[q1_reverse_order_indx]
        pad_q1_back_orig = pad_q1_back[q1_reverse_order_indx]
        h_q1_forward_orig = h_q1_forward[q1_reverse_order_indx]
        h_q1_back_orig = h_q1_back[q1_reverse_order_indx]
        q1_lens_orig = q1_lens[q1_reverse_order_indx]
        ##Q2
        q2_out,q2_hidden = self.crl(q2_pack_pad_seq_embed)
        pad_q2_out = pad_packed_sequence(q2_out, batch_first=True)
        q2h,_ = q2_hidden
        pad_q2_forward,pad_q2_back = torch.chunk(pad_q2_out[0],2,dim=2)##NLC
        h_q2_forward = q2h[-2]
        h_q2_back = q2h[-1]
        pad_q2_forward_orig = pad_q2_forward[q2_reverse_order_indx]
        pad_q2_back_orig = pad_q2_back[q2_reverse_order_indx]
        h_q2_forward_orig = h_q2_forward[q2_reverse_order_indx]
        h_q2_back_orig = h_q2_back[q2_reverse_order_indx]
        q2_lens_orig = q2_lens[q2_reverse_order_indx]

        q1_for_full_matching = self.full_matching(pad_q1_forward_orig,h_q2_forward_orig,self.MW1)
        q1_back_full_matching = self.full_matching(pad_q1_back_orig,h_q2_back_orig,self.MW2)
        q1_for_maxpool_matching = self.maxpool_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW3,q2_lens_orig)
        q1_back_maxpool_matching = self.maxpool_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW4,q2_lens_orig)
        q1_for_att_matching = self.attentive_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW5)
        q1_back_att_matching = self.attentive_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW6)
        q1_for_maxatt_matching = self.max_attentive_matching(pad_q1_forward_orig,pad_q2_forward_orig,self.MW7,q2_lens_orig)
        q1_back_maxatt_matching = self.max_attentive_matching(pad_q1_back_orig,pad_q2_back_orig,self.MW8,q2_lens_orig)

        q2_for_full_matching = self.full_matching(pad_q2_forward_orig,h_q1_forward_orig,self.MW1)
        q2_back_full_matching = self.full_matching(pad_q2_back_orig,h_q1_back_orig,self.MW2)
        q2_for_maxpool_matching = self.maxpool_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW3,q1_lens_orig)
        q2_back_maxpool_matching = self.maxpool_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW4,q1_lens_orig)
        q2_for_att_matching = self.attentive_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW5)
        q2_back_att_matching = self.attentive_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW6)
        q2_for_maxatt_matching = self.max_attentive_matching(pad_q2_forward_orig,pad_q1_forward_orig,self.MW7,q1_lens_orig)
        q2_back_maxatt_matching = self.max_attentive_matching(pad_q2_back_orig,pad_q1_back_orig,self.MW8,q1_lens_orig)

        q1_agg = torch.cat([q1_for_full_matching,
                            q1_back_full_matching,
                            q1_for_maxpool_matching,
                            q1_back_maxpool_matching,
                            q1_for_att_matching,
                            q1_back_att_matching,
                            q1_for_maxatt_matching,
                            q1_back_maxatt_matching
                             ],dim=2) ##NXLX8P
        #print("q1_agg")
        #print(q1_agg.size())
        q2_agg = torch.cat([
            q2_for_full_matching,
            q2_back_full_matching,
            q2_for_maxpool_matching,
            q2_back_maxpool_matching,
            q2_for_att_matching,
            q2_back_att_matching,
            q2_for_maxatt_matching,
            q2_back_maxatt_matching
        ],dim=2)##NXLX8P
        #print("q2_agg")
        #print(q2_agg.size())
        q1_agg_order = q1_agg[q1_perm_idx]
        q2_agg_order = q2_agg[q2_perm_idx]
        q1_pack_pad_agg_order = pack_padded_sequence(q1_agg_order, batch_first=True, lengths=q1_lens)
        q2_pack_pad_agg_order = pack_padded_sequence(q2_agg_order, batch_first=True, lengths=q2_lens)

        q1_agout,q1_aghidden = self.al(q1_pack_pad_agg_order)
        q1agh,_ = q1_aghidden

        q2_agout,q2_aghidden = self.al(q2_pack_pad_agg_order)
        q2agh,_ = q2_aghidden


        q1_agencode = torch.cat((q1agh[-2], q1agh[-1]), dim=1)
        q2_agencode = torch.cat((q2agh[-2], q2agh[-1]), dim=1)
        q1_encode_reverse = q1_agencode[q1_reverse_order_indx]
        q2_encode_reverse = q2_agencode[q2_reverse_order_indx]
        q_pair_encode_q12= torch.cat((q1_encode_reverse,q2_encode_reverse),dim=1)
        hid1 = self.linear1_drop(F.relu(self.linear1(q_pair_encode_q12)))
        hid2 = self.linear2_drop(F.relu(self.linear2(hid1)))
        out = self.linear3(hid2)
        return out
```
##### Ensemble
Finally ensemble is adopted to achieve the best score.
