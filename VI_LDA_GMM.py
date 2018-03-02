### 1) No periodic savings
### 2) split randomly -- to test stability of using perplexity as the metric
### 3) use quantile normalized features

import numpy as np
import pandas as pd
import time
import pickle
import copy

from collections import Counter
from scipy.special import digamma
from scipy.stats import gamma
from scipy.special import gammaln
from scipy.special import multigammaln
import time

import sys
import os
import pdb


class LDA_GMM:

    def __init__(self, K, alphai, var_nu, var_lbd, episron):

	# store genes
	self.genes = [x.replace('_normed_features.txt','') for x in files]
        self.test_genes = np.random.choice(self.genes, int(len(self.genes) * test_proportion), replace=False)

        self.Nd = []
        self.trainWn = []
        self.trainWn_names = []  ### should be aligned to self.trainWn
        self.testWn = []
        self.testWn_names = []

	for fn in files: 
		g = fn.replace('_normed_features.txt','')
        	#### store the data
		feature_one_gene = pd.read_csv('%s/feature_matrix/%s' % (datadir, fn), sep='\t', index_col = 0)
		feature_one_gene = feature_one_gene.drop_duplicates()
		if PERMUTE:
			temp = feature_one_gene.values
			np.random.shuffle(temp)
			temp = temp.T
			np.random.shuffle(temp)
			feature_one_gene.value = temp.T
		feature_one_gene['intercept'] = np.ones(len(feature_one_gene))
		self.vocabulary = np.sort(feature_one_gene.columns)
		feature_one_gene = feature_one_gene[self.vocabulary]
		gnumber = len(feature_one_gene)
		if g in self.test_genes:
			# train and test regions
			test_idx = np.random.choice(feature_one_gene.index, int(gnumber * 0.1), replace = False)
			train_idx = [x for x in feature_one_gene.index if x not in test_idx]
			gnumber = len(train_idx)
			# store the regions
			self.trainWn.append(np.array(feature_one_gene.loc[train_idx]))
			self.trainWn_names.append(train_idx)
			self.testWn.append(np.array(feature_one_gene.loc[test_idx]))
			self.testWn_names.append(test_idx)
		else:
			self.trainWn.append(np.array(feature_one_gene))
			self.trainWn_names.append(feature_one_gene.index)
		self.Nd.append(gnumber)

        self.K = K
        self.M = len(self.genes)
        self.V = len(self.vocabulary)
        
        
        #### super-parameters
	self.alphai = alphai
	self.m0 = np.mean(np.concatenate(self.trainWn), axis=0)
        self.beta0 = var_nu ## variance of the mean vector
        self.W0 = np.identity(self.V) * 1.0
        self.nu0 = self.V + var_lbd ## degree of freedom 

	self.episron = episron

	print 'Parameter settings:'
	print 'M = %d, V = %d, K = %d' % (self.M, self.V, self.K)
	print 'alphi = ', self.alphai, 'beta0 = ', self.beta0, 'nu0 = ', self.nu0, 'episron = ', self.episron
	print 'test_proportion = ', test_proportion
	print 'PERMUTED = ', PERMUTE


    def initialize(self):        
        
        self.W0_inv = np.linalg.inv(self.W0)
        
        #### randomly initialize phi 
        self.phi = []
        for d in xrange(self.M):
            self.phi.append(vector_sum_to_1(self.Nd[d], self.K))
            self.phi[d] = np.array(map(lambda x: (x+1e-300) / np.sum(x+1e-300), self.phi[d]))
            
        ## Define some statistics for convinence
        # Nk
	self.Nk = np.sum(np.concatenate(self.phi), axis=0)
        # weighted mean value in each module
	self.wk_bar = np.dot(np.transpose(np.concatenate(self.phi)), np.concatenate(self.trainWn))
        #self.wk_bar = np.array([sum(l) for l in zip(*[np.dot(np.transpose(self.phi[t]), self.trainWn[t]) for t in xrange(self.M)])])
        self.wk_bar = np.array([self.wk_bar[k,:] / self.Nk[k] for k in xrange(self.K)])
        # weighted covariance
        self.Sk = []
        for k in xrange(self.K):
            centered_Wnd = np.concatenate(self.trainWn) - self.wk_bar[k,:][np.newaxis]
            self.Sk.append(np.dot(np.concatenate(self.phi)[:,k][np.newaxis] * np.transpose(centered_Wnd), centered_Wnd) / self.Nk[k])

            
        #### Initialization the other parameters
        # Lambda
        self.betak = self.beta0 + self.Nk
        self.nuk   = self.nu0 + self.Nk
        self.mk    = [(self.beta0 * self.m0 + self.Nk[k] * self.wk_bar[k,:]) / (self.beta0 + self.Nk[k]) for k in xrange(self.K)]
        self.Wk    = []
        for k in xrange(self.K):
            self.Wk.append(self.W0_inv + self.Nk[k] * self.Sk[k] + \
                            self.beta0 * self.Nk[k] / (self.beta0 + self.Nk[k]) * \
                            np.dot((self.wk_bar[k,:] - self.m0)[np.newaxis].T, (self.wk_bar[k,:] - self.m0)[np.newaxis]))
            if np.linalg.cond(self.Wk[k]) > 1/sys.float_info.epsilon:
                    print 'Wk_%d is not psd at initialization ' % (k)
            else:
                    self.Wk[-1] = np.linalg.inv(self.Wk[-1])
        assert sum([np.linalg.slogdet(self.Wk[k])[0] < 1 for k in xrange(self.K)]) == 0
        
        self.Eq_lnDetLambda = np.zeros(self.K)
        for k in xrange(self.K):
            self.Eq_lnDetLambda[k] = np.sum([digamma((self.nuk[k] + 1 - j)/2) for j in xrange(1,self.V+1)]) + \
                                     self.V * np.log(2) + np.linalg.slogdet(self.Wk[k])[1]        
        
        # gammma
        self.gammma = [np.sum(self.phi[d], axis=0) + self.alphai for d in xrange(self.M)]
        self.gammma = np.array(self.gammma)
                
        #### storage
        self.ELBO = []
        self.T = []
        self.updated = []        
        self.compute_ELBO('Initial')


    def update_phi(self):
            for d in xrange(self.M):
                ## [Nd, K], every row is the same for each matrix.
                Eq_mu_Lambda_mean = []
                for k in xrange(self.K):
                    wk_minus_mk = self.trainWn[d] - self.mk[k][np.newaxis]
                    Eq_mu_Lambda_mean.append(np.diag(self.V / self.betak[k] + \
                                              self.nuk[k] * np.dot(np.dot(wk_minus_mk, self.Wk[k]), np.transpose(wk_minus_mk))))
                Eq_mu_Lambda_mean = np.transpose(Eq_mu_Lambda_mean)
                self.Eq_mu_Lambda_mean = Eq_mu_Lambda_mean
                # scale to avoid overflow   
                constant_to = np.max(digamma(self.gammma[d,:])[np.newaxis] + (self.Eq_lnDetLambda[np.newaxis] - self.Eq_mu_Lambda_mean)/2, axis=1)
                self.phi[d] = np.exp(digamma(self.gammma[d,:])[np.newaxis] + self.Eq_lnDetLambda[np.newaxis]/2 - self.Eq_mu_Lambda_mean/2 - np.transpose(constant_to[np.newaxis]))
                self.phi[d] = np.array(map(lambda x: (x+1e-300) / np.sum(x+1e-300), self.phi[d]))
            ### Define some statistics for convinence
            # Nk
            self.Nk = np.sum(np.concatenate(self.phi), axis=0)
            # weighted mean value in each module
            self.wk_bar = np.dot(np.transpose(np.concatenate(self.phi)), np.concatenate(self.trainWn)) / (self.Nk[np.newaxis].T)
            # weighted covariance
            self.Sk = []
            for k in xrange(self.K):
                centered_Wnd = np.concatenate(self.trainWn) - self.wk_bar[k,:][np.newaxis]
                self.Sk.append(np.dot(np.concatenate(self.phi)[:,k][np.newaxis] * np.transpose(centered_Wnd), centered_Wnd) / self.Nk[k])
	    #self.compute_ELBO('phi')




    def update_gamma(self):
            self.gammma = [np.sum(self.phi[d], axis=0) + self.alphai for d in xrange(self.M)]
            self.gammma = np.array(self.gammma)
            #self.compute_ELBO('gamma')




    def update_lmda(self):
            self.betak = self.beta0 + self.Nk
            self.mk = [(self.beta0 * self.m0 + self.Nk[k] * self.wk_bar[k,:]) / (self.beta0 + self.Nk[k]) for k in xrange(self.K)]
            self.nuk = self.nu0 + self.Nk
            for k in xrange(self.K):
                self.Wk[k] = self.W0_inv + self.Nk[k] * self.Sk[k] + \
                            self.beta0*self.Nk[k] / (self.beta0+self.Nk[k]) * \
                            np.dot((self.wk_bar[k,:] - self.m0)[np.newaxis].T, (self.wk_bar[k,:] - self.m0)[np.newaxis])
                if np.linalg.cond(self.Wk[k]) > 1/sys.float_info.epsilon:
                    print 'Wk_%d is not psd at iteration %d ' % (k,iterations)
                    return
                else:
                    self.Wk[k] = np.linalg.inv(self.Wk[k])

            for k in xrange(self.K):
                self.Eq_lnDetLambda[k] = np.sum([digamma((self.nuk[k] + 1 - j)/2) for j in xrange(1,self.V+1)]) + \
                                         self.V * np.log(2) + np.linalg.slogdet(self.Wk[k])[1]
            #self.compute_ELBO('Lambdas')

	
    def update(self, max_iter):
        
        iterations = 0
        while iterations < max_iter: 
            START = time.time()

            ### update phi
            self.update_phi() 
            ### update gammma
	    self.update_gamma()
            ### update lambda
	    self.update_lmda()
                   
	    self.compute_ELBO(iterations) 
            assert sum([np.linalg.slogdet(self.Wk[k])[0] < 1 for k in xrange(self.K)]) == 0
            assert ~np.isnan(sum([np.sum(self.phi[t]) for t in xrange(self.M)]))
            assert ~np.isnan(np.sum(self.gammma))
            assert ~np.isnan(np.sum(self.Wk))
        
            #print Counter([a for b in [np.argmax(self.phi[d], axis=1) for d in xrange(self.M)] for a in b])
	    self.T.append(time.time() - START)   

            if self.ELBO[-1] - self.ELBO[-2] < self.episron:
                print 'Converged after %d iterations' % iterations
                break
            else:
                iterations += 1


 
    def compute_ELBO(self, qi_updated):
                        
	y = gammaln(np.sum(self.gammma, axis=1))[np.newaxis]
        gammma_term = np.sum(gammaln(self.gammma) - np.transpose(y) )
        phi_term = -np.sum([np.sum(t * np.log(t)) for t in self.phi])

        mu_lambda_term = []
        for k in xrange(self.K):
            temp1 = (self.Nk[k] + self.nu0 - self.V - 1) * self.Eq_lnDetLambda[k]
            temp2 = self.V * (self.Nk[k] + self.beta0) / self.betak[k]
            temp4 = self.nuk[k] * (self.Nk[k]  *  np.dot(np.dot(self.wk_bar[k] - self.mk[k], self.Wk[k])[np.newaxis], \
                                                         (self.wk_bar[k] - self.mk[k])[np.newaxis].T) + \
                                        self.beta0 * np.dot(np.dot(self.mk[k] - self.m0, self.Wk[k])[np.newaxis], \
                                                        (self.mk[k] - self.m0)[np.newaxis].T))
            temp4 = temp4[0,0]
            temp5 = self.nuk[k] * (self.Nk[k] * np.matrix.trace(np.dot(self.Sk[k], self.Wk[k])) + \
                                    np.matrix.trace(np.dot(self.W0_inv, self.Wk[k])))
            temp6 = self.V * self.Nk[k] * np.log(2*np.math.pi) + self.V * np.log(self.betak[k]/(2*np.math.pi)) 
            H_q_Lambdak = (self.V+1)/2 * np.linalg.slogdet(self.Wk[k])[1] + \
                            multigammaln(self.nuk[k]/2, self.V) - \
                            (self.nuk[k]-self.V-1)/2 * np.sum([digamma((self.nuk[k]-j+1)/2) for j in xrange(1, self.V+1)]) + \
                            self.nuk[k] * self.V / 2
            mu_lambda_term.append(temp1 - temp2 - temp4 - temp5 - temp6 + 2*H_q_Lambdak)
            
        ELBO = gammma_term + phi_term + np.sum(mu_lambda_term) / 2
        self.ELBO.append(ELBO)

	if len(self.ELBO) > 1:
		assert (self.ELBO[-1] - self.ELBO[-2]) > 0

        #self.updated.append(qi_updated)
        #print qi_updated, ELBO
    


    def compute_llog(self):
	n_genes = len(self.test_genes)

	# compute some statistics to avoid repeated work
        self.precision = [] 
        self.logPrecision = np.zeros(self.K)
        for k in xrange(self.K):
                self.precision.append(self.nuk[k] * self.Wk[k])
                self.logPrecision = np.linalg.slogdet(self.precision)[1]

	# in training data
        train_genes = np.random.choice(list(set(self.genes) - set(self.test_genes)), n_genes, replace=False)
	# probability of every region given the gene
	words_d = np.array(self.trainWn)[np.where([(x in train_genes) for x in self.genes])[0]]
	train_Nd = [len(x) for x in words_d]
	words_d = np.concatenate(words_d)
	pw_d = np.array([self.mvn_log(words_d, k) for k in xrange(self.K)])
	pw_d = np.transpose(pw_d)
        # cluster distribution for each gene
        train_gammma = self.gammma[np.where([(x in train_genes) for x in self.genes])[0]]
        train_theta = train_gammma / np.transpose(np.sum(train_gammma, axis=1)[np.newaxis])
        train_theta = [np.reshape(np.repeat(train_theta[x], train_Nd[x]), [train_Nd[x],self.K]) for x in xrange(len(train_Nd))]
        train_theta = np.concatenate(train_theta)
        assert len(train_theta) == len(words_d)
        # probability of every region
        pw = pw_d * train_theta
        pw_each_region = np.sum(pw, axis=1)
        logpw = np.sum(np.log(pw_each_region))
	self.log_perplexity = [ -logpw / np.sum(train_Nd)]

	# in test data
    	test_genes = np.random.choice(self.test_genes, n_genes, replace=False)
	# probability of every region given the gene
        words_d = np.array(self.testWn)[np.where([(x in test_genes) for x in self.test_genes])[0]]
	test_Nd = [len(x) for x in words_d]
        words_d = np.concatenate(words_d)
        pw_d = np.array([self.mvn_log(words_d, k) for k in xrange(self.K)])
	pw_d = np.transpose(pw_d)
	# cluster distribution for each gene
	test_gammma = self.gammma[np.where([(x in test_genes) for x in self.genes])[0]]
	test_theta = test_gammma / np.transpose(np.sum(test_gammma, axis=1)[np.newaxis])	
	test_theta = [np.reshape(np.repeat(test_theta[x], test_Nd[x]), [test_Nd[x],self.K]) for x in xrange(len(test_Nd))]
	test_theta = np.concatenate(test_theta)
	assert len(test_theta) == len(words_d)
	# probability of every region
        pw = pw_d * test_theta
        pw_each_region = np.sum(pw, axis=1)
        logpw = np.sum(np.log(pw_each_region))
        self.log_perplexity.append( -logpw / np.sum(test_Nd))

	print 'Log perplexity in training data = %.2f, in test data = %.2f' % (self.log_perplexity[0], self.log_perplexity[1])

	
    def delWn(self):
	self.trainWn = []        
    
    

    def mvn_log(self, x, k):
        # mu.shape: (D,)
        llk = 0.5 * self.logPrecision[k] - np.diag(0.5 * np.dot(np.dot((x-self.mk[k]), self.precision[k]), (x-self.mk[k]).T)) 
	lk = np.exp(llk)
        return lk


def vector_sum_to_1(rowN, colN):
    ## row sum up to 1
    vc = np.random.sample([rowN, colN])
    vc = map(lambda x: x / np.sum(x), vc)
    return np.array(vc)



def main():        

    OBJECT = LDA_GMM(Moduel_K, alphai, var_nu, var_lbd, episron)

    S = time.time()
    OBJECT.initialize()
    print 'Initialization takes %f s' % (time.time() - S)

    S = time.time()
    OBJECT.update(max_iter)
    print 'Update takes %f s' % (time.time()-S)

    S = time.time()
    OBJECT.compute_llog()
    print 'Compute llog takes %f s' % (time.time()-S)

    ## save
    OBJECT.delWn()
    with open(output, 'wb') as f:
        pickle.dump(OBJECT, f)






if __name__=='__main__':
	datadir = '/work-zfs/abattle4/heyuan/LDA_GMM/featureSet'
	outdir = '/work-zfs/abattle4/heyuan/LDA_GMM/results'

	Moduel_K  = int(sys.argv[1])
	alphai = float(sys.argv[2])
	var_nu = float(sys.argv[3])
	var_lbd = float(sys.argv[4])
	episron = float(sys.argv[5])
	test_proportion = float(sys.argv[6])

	max_iter = int(sys.argv[7])
	PERMUTE  = int(sys.argv[8])
	randomID = int(sys.argv[9])


	files = [f for f in os.listdir('%s/feature_matrix' % datadir) if f.endswith('_normed_features.txt')]
	files = np.sort(files)
	output = '%s/K_%d_varnu_%d_varlbd_%d_alphai_%s_epi_%s_testPro_%s_maxIter_%d_randSplit_%d_permute%d.p' % (outdir, Moduel_K, int(var_nu), int(var_lbd), str(alphai), str(episron), str(test_proportion), max_iter, randomID, PERMUTE)

	main()









