import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import gammaln
import time

import pickle


# In[2]:


class LDA:
    
    def __init__(self, fn, K):
        data = pd.read_csv(fn,sep='\t',index_col = 0)
        np.random.seed(2)
        data = data.iloc[np.random.choice(range(len(data)),5000)]
        words_in_each_gene = data.groupby(data.index).sum()

        words_in_txt = [np.repeat(np.array(words_in_each_gene.columns), t.astype('int'),axis=0) for t in np.array(np.ceil(words_in_each_gene))]
        allwords = [a for b in words_in_txt for a in b]
        vocabulary = np.unique(allwords)
        vocabulary_in_int = dict(zip(vocabulary, range(len(vocabulary))))
        self.vocabulary_in_int = vocabulary_in_int

        self.K = K
        self.M = len(words_in_each_gene)
        self.V = len(vocabulary)
        print 'M = %d, V = %d, total number of words = %d' % (self.M, self.V, len(allwords))

        W = [[vocabulary_in_int[t] for t in wd] for wd in words_in_txt]
        self.Nd = [len(wd) for wd in W]
        self.alphii = 1/float(self.K)
        self.etaj = 1/float(self.V)
        
        ### generate W matrix: D x [Nd, V]
        self.Wn = [np.zeros([self.Nd[t],self.V]) for t in range(self.M)]
        for d in range(self.M):
            for token in xrange(self.Nd[d]):
                self.Wn[d][token, W[d][token]] = 1 
          
        
        #### initialization -- set the strucutres, values don't matter
        self.phi = [np.ones([t, self.K]) for t in self.Nd]
        self.gammma = np.random.sample([self.M, self.K])
        self.lambbda = np.random.sample([self.K, self.V])

        
    def update_stochastic(self, max_iter, local_max_iter):
        
        self.T = []
        self.ELBO = []
        episron = 1e-3
        iterations = 1
        old_ELBO = 0

        while iterations < max_iter: 
    
            START = time.time()
            
            ##### update local parameters
            
            ## shuffle the samples
            shutffled_idx = range(self.M)
            np.random.shuffle(shutffled_idx)
            for rd_d in shutffled_idx:
                print rd_d
                nd_d = self.Nd[rd_d]
            
                # randomly initialize gamma for this gene
                # & give phi_local a strucutre
                gammma_local = np.random.sample(self.K)
                phi_local = np.zeros([rd_d, self.K])
            
                # optimize the local phi and gamma for this document
                local_iterations = 0
                while local_iterations < local_max_iter:
                
                    old_phi_local = self.phi[rd_d].copy()
                    old_gammma_local = self.gammma[rd_d].copy()
                
                    # update local phi
                    A = np.transpose(np.reshape(np.repeat(digamma(gammma_local), nd_d), [self.K, nd_d]))
                    B = np.dot(self.Wn[rd_d], digamma(np.transpose(self.lambbda)))
                    C = np.transpose(digamma(np.reshape(np.repeat(np.sum(np.transpose(self.lambbda), axis=0), nd_d), [self.K, nd_d]))) 
                
                    phi_local = A+B-C
                    phi_local = np.exp([x - max(x) for x in phi_local])
                    phi_local = np.array([x+1e-10 / sum(x+1e-10) for x in phi_local])
                              
                    # update local gamma 
                    gammma_local = np.sum(phi_local, axis=0) + self.alphii
                    if np.sum(np.linalg.norm(old_phi_local - phi_local) + np.linalg.norm(old_gammma_local - gammma_local)) < episron:
                        print 'Gene converged after %d iterations' % local_iterations
                        break
                    else:
                        self.phi[rd_d] = phi_local
                        self.gammma[rd_d] = gammma_local
                        local_iterations += 1
                                            
                assert ~np.isnan(np.sum(self.phi[rd_d]))
                assert np.sum(self.phi[rd_d] == 0) == 0
                assert ~np.isnan(np.sum(self.gammma[rd_d]))
                                
                    
            ##### update global parameters
            
            kappa = 0.55
            rhot = np.power(iterations + 1, kappa)
            
            ## update intermediate global parameter
            lambbda_med = np.zeros([self.K, self.V])
            for i in xrange(self.K):
                for j in xrange(self.V):
                    lambbda_med[i,j] = self.M * np.dot(self.Wn[rd_d][:,j], phi_local[:,i]) + self.etaj

            self.lambbda = (1-rhot) * self.lambbda  + rhot * lambbda_med
            assert ~np.isnan(np.sum(self.lambbda))


            ##### compute ELBO
            gammma_term = np.sum(np.array([gammaln(np.sum(t)) for t in self.gammma]) - np.sum([gammaln(t) for t in self.gammma],axis=1))
            lambda_term = np.sum(np.array([gammaln(np.sum(t)) for t in self.lambbda]) - np.sum([gammaln(t) for t in self.lambbda],axis=1))
            phi_term = np.sum([np.sum(t * np.log(t)) for t in self.phi])
            ELBO = -gammma_term - lambda_term - phi_term
            self.ELBO.append(ELBO)
            self.T.append(time.time() - START)
    
            print time.time() - START
            print gammma_term, lambda_term, phi_term, ELBO
    
            if np.abs(ELBO - old_ELBO) < episron:
                print 'Converged after %d iterations\n' % iterations
                break
            else:
                iterations += 1
                old_ELBO = ELBO.copy()
                
                
            


if __name__ == '__main__':
    genes = LDA("Features.csv",4)
    genes.update_stochastic(20,1000)

    output = "VI/VI_stochatic_K4"

    np.savez('%s-ELBO' %output, genes.ELBO)
    np.savez('%s-t' % output, genes.T)
    np.savez('%s-gamma' % output, genes.gammma)
    np.savez('%s-lambda' % output, genes.lambbda)
    np.savez('%s-phi' % output, genes.phi)
    np.savez('%s-data-used' % output, genes.Wn)
    np.savez('%s-vocabulary-to-int' % output, genes.vocabulary_in_int)

    with open("%s.pickle" % output, 'wb') as handle:
        pickle.dump(genes, handle)


