import json
import numpy as np
import pystan

def gen_words(docs):
    word_list = []
    doc_list = []
    for i, doc in enumerate(docs):
        word_list += [j + 1 for j in doc]
        doc_list += [i + 1 for j in range(len(doc))]
    return word_list, doc_list

if __name__ == '__main__':
    with open('docs.txt') as reader:
        docs = json.loads(reader.read())
    word_arr, doc_arr = gen_words(docs)
    lda_data = {
        'N': len(word_arr),
        'V': len(set(word_arr)),
        'M': len(set(doc_arr)),
        'K': 3,
        'alpha': [0.15, 0.05, 0.15],
        'beta': [0.1, 0.1, 0.1, 0.1, 0.1],
        'words': word_arr,
        'docs': doc_arr
    }
    sm = pystan.StanModel(file='lda.stan')
    fit = sm.sampling(data=lda_data, iter=1000, chains=3)
    
    