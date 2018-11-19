import json
import numpy as np

def gen_words(docs):
    word_list = []
    doc_list = []
    for i, doc in enumerate(docs):
        word_list += doc
        doc_list += [i for i in range(len(doc))]
    return np.array(word_list), np.array(doc_list)

if __name__ == '__main__':
    with open('docs.txt') as reader:
        docs = json.loads(reader.read())
    word_arr, doc_arr = gen_words(docs)
    print(word_arr.shape, doc_arr.shape)
    