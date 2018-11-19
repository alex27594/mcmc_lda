import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def visualize_distr(samples):
    x, y, z = [samples[:, i] for i in range(3)]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xs=x, ys=y, zs=z)
    plt.show()

def interprete_doc(doc, words):
    return ' '.join([words[i] for i in doc])

def interprete_docs(docs, docs_topics, words):
    for i in range(len(docs)):
        print('Topics: {} {}, {} {}, {} {}'.format(
            *sum(
                    zip(topics, docs_topics[i]),
                    ()
                )
            )
        )
        print(interprete_doc(docs[i], words))


def generate(n_docs, n_doc_words, word_topic_matrix, alpha):
    n_topics = word_topic_matrix.shape[0]
    n_words = word_topic_matrix.shape[1]
    docs = []
    docs_topics = []
    for i in range(n_docs):
        doc= []
        doc_topics = stats.dirichlet.rvs(alpha=alpha, size=1).reshape(-1,)
        for j in range(n_doc_words):
            topic = int(np.random.choice(a=list(range(n_topics)), size=1, p=doc_topics)[0])
            word = int(np.random.choice(a=list(range(n_words)), size=1, p=word_topic_matrix[topic, :].reshape(-1,))[0])
            doc.append(word)
        docs.append(doc)
        docs_topics.append(doc_topics)
    return docs, docs_topics

if __name__ == '__main__':
    alpha = [2, 2, 2]
    topics = ['war', 'science', 'love']
    words = ['war', 'science', 'warrior', 'analysis', 'love']
    word_topic_matrix = np.array([
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.7, 0.0, 0.3, 0.0],
    [0.0, 0.1, 0.1, 0.0, 0.8]])
    n_docs = 1000
    # docs, docs_topics = generate(n_docs=10, n_doc_words=10, 
    #                             word_topic_matrix=word_topic_matrix,
    #                             alpha=alpha)
    # with open('docs.txt', 'w') as writer:
    #     obj = json.dumps(docs)
    #     writer.write(obj)    
    samples = stats.dirichlet.rvs(alpha=[1, 1, 0.1], size=1000)
    visualize_distr(samples)





