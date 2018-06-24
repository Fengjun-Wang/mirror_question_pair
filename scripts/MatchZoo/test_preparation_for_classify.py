# /bin/python2.7
import os
import sys
import numpy as np
# sys.path.append('../../matchzoo/inputs')
# sys.path.append('../../matchzoo/utils')
from preparation import *
from preprocess import *


def main(mode, words_or_chars):
    basedir = '../../input/matchzoo/{0}_{1}'.format(mode, words_or_chars)
    save_txt_file_name = 'sample.txt'
    try:
        os.makedirs(basedir)
        print 'created folder: ', basedir, 'check it!'
        exit(1)
    except:
        pass
    save_txt_path = os.path.join(basedir, save_txt_file_name)

    # transform query/document pairs into corpus file and relation file
    prepare = Preparation()
    corpus, rels = prepare.run_with_one_corpus(save_txt_path)
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(os.path.join(basedir, 'corpus.txt'), corpus)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.4, 0.3, 0.3])
    prepare.save_relation(os.path.join(basedir, 'relation_train.txt'), rel_train)
    prepare.save_relation(os.path.join(basedir, 'relation_valid.txt'), rel_valid)
    prepare.save_relation(os.path.join(basedir, 'relation_test.txt'), rel_test)
    print('preparation finished ...')

    # Prerpocess corpus file
    preprocessor = Preprocess()

    dids, docs = preprocessor.run(os.path.join(basedir, 'corpus.txt'))
    preprocessor.save_word_dict(os.path.join(basedir, 'word_dict.txt'))
    preprocessor.save_words_stats(os.path.join(basedir, 'word_stats.txt'))

    fout = open(os.path.join(basedir, 'corpus_preprocessed.txt'),'w')
    for inum,did in enumerate(dids):
        fout.write('%s %d %s\n'%(did, len(docs[inum]), ' '.join(map(str,docs[inum]))))
    fout.close()
    print('preprocess finished ...')

if __name__ == '__main__':
    # for mode in ['train', 'test']:
    #     for words_or_chars in ['words', 'chars']:
    for mode in ['test']:
        for words_or_chars in ['chars', 'words']:
            main(mode=mode, words_or_chars=words_or_chars)