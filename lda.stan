data {
    int<lower=1> N; // number of words
    int<lower=2> V; // number of unique words
    int<lower=1> M; //number of docs
    int<lower=2> K; // number of topics
    vector<lower=0>[K] alpha; // parameter of topics prior Dirichlet distribution
    vector<lower=0>[V] beta; // parameter of words prior Dirichlet distribution
    int<lower=1, upper=V> words[N]; // all words
    int<lower=1, upper=M> docs[N]; // doc indices of all words
}
parameters {
    simplex[K] theta[M]; // discrete distribution over topics for all docs
    simplex[V] phi[K]; // discrete distribution over words for all topics
}
model {
    for (m in 1:M) {
        theta[m] ~ dirichlet(alpha);
    }
    for (k in 1:K) {
        phi[k] ~ dirichlet(beta);
    }
    for (n in 1:N) {
        real gamma[K];
        for (k in 1:K) {
            gamma[k] = log(phi[k, words[n]]) + log(theta[docs[n], k]);
        }
        target += log_sum_exp(gamma);
    }
}