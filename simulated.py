### Run on HHPC

import sys
import numpy as np
import pandas as pd
import time
import dill

## for simulation
from numpy.random import dirichlet
from numpy.random import multinomial
from numpy.random import multivariate_normal
from scipy.stats import wishart



class LDA_GMM:

    def __init__(self, alphai, M, K, V):

        ### Since wishart cannot be run on local machine, need to generate the Lambdas from HHPC


        ### What needs to be stored: 
        #   1) the observed variables -- for inference: M, Nd, K, V, Wn
        #   2) true modules -- for evaluation: word_assginment_z, theta
        #   3) superparameters -- for better inference: alphai, nu0, W0, m0, beta0

        self.alphai = alphai
        self.M = M
        self.K = K
        self.V = V
        np.random.seed(2)
        self.Nd = np.random.choice(range(150,300),M)   ## number of regions in each gene

        ## cluster assignment for each region
        self.theta = dirichlet(np.ones(K)*alphai, M)
        word_assginment_z = []
        for d in xrange(M):
            word_assginment_z.append([multinomial(1, self.theta[d]) for t in xrange(self.Nd[d])])
            word_assginment_z[d] = np.array(word_assginment_z[d])

        ## modules
        self.nu0 = V + 5.0
        self.m0 = np.zeros(V)
        self.beta0 = 1.0
        self.W0 = np.identity(V)

        Lambbda = wishart.rvs(df=self.nu0, scale=self.W0, size=K, random_state=0)
        Sigma = [np.linalg.inv(lll) for lll in Lambbda]
        mu = [multivariate_normal(self.m0, Sigma[k]/self.beta0, size=1)[0] for k in xrange(K)]

        ## observations
        words_values = []
        for d in xrange(M):
            word_assginment_z_for_this_d = [np.where(zzz)[0][0] for zzz in word_assginment_z[d]]
            words_values.append([multivariate_normal(mu[zzz], Sigma[zzz], size = 1)[0] for zzz in word_assginment_z_for_this_d])
            words_values[d] = np.array(words_values[d])
        self.Wn = words_values
        self.word_assginment_z = word_assginment_z

        print np.std(np.concatenate(words_values))
        print np.linalg.det(np.cov(np.transpose(np.concatenate(words_values))))
        print np.linalg.cond(np.cov(np.transpose(np.concatenate(words_values))))

if __name__=='__main__':
    	K = int(sys.argv[1])
    	OBJECT = LDA_GMM(0.25, 200, K, 150)
    	dill.dump(OBJECT, open('/scratch1/battle-fs1/heyuan/LDA_GMM/simulate/K_%d.p' % K, "wb" ) )


