import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
import dill
import seaborn as sns


from scipy.spatial import distance
from scipy.cluster import hierarchy



def plot_results(output):

    ## read in 
    with open(output, 'rb') as fn:
        result = dill.load(fn)

    ## extract ELBO
    ELBO = result.ELBO
    time_used = np.array(result.T)
    time_used = time_used / 60 / 60
    title_text = 'Converged in %d iterations, %f h' % (len(time_used), np.sum(time_used))

    AA = [result.ELBO[0]]
    temp = result.ELBO[1:]

    for k in xrange(len(temp)):
        if k%3 == 2:
            AA.append(temp[k])
    print np.array([j-i for i, j in zip(AA[:-1], AA[1:])])[np.array([j-i for i, j in zip(AA[:-1], AA[1:])]) < 0]
    
    plt.figure()
    plt.plot(range(len(AA)), AA)
    plt.xlabel('Iterations')
    plt.ylabel('ELBO')
    plt.title(title_text)
    plt.savefig('%s_ELBO.png' % output)
    plt.close()
    
    # increase_ELBO = [j-i for i, j in zip(result.ELBO[:-1], result.ELBO[1:])]
    # x = pd.DataFrame(zip(result.updated[1:], increase_ELBO))
    # print x[x[1] < 0]


    # Distribution of modules across genes
    Theta = pd.DataFrame(result.gammma)
    Theta.index = result.genes
    Theta = Theta.apply(lambda x: x / np.sum(x), axis=1)
    
    correlations_array = np.asarray(Theta)
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')

    sns.set(font_scale=0.5)
    sns.clustermap(Theta, row_linkage=row_linkage, col_cluster=False, method="average")
    plt.savefig('%s_Theta.png' % output)
    plt.close()

    # print 'Clustered genes based on Theta'
    # gene_clusters = pd.DataFrame(zip(hierarchy.fcluster(row_linkage, 1.15468), result.genes))
    # gene_clusters.columns = ['Cluster_ID', 'geneID']
    # for df_slice in gene_clusters.groupby(gene_clusters['Cluster_ID']):
    #     print np.array(df_slice[1]['geneID'])
        
        
    # phi
    
    if 0:
        '''
        Too slow, no need to compute every time
        '''
        d = 0
        phi_dist = pd.DataFrame.from_dict(dict(Counter(np.argmax(result.phi[d], axis=1))), orient='index') / len(result.phi[d])
    
        for d in xrange(1, result.M):
            phi_d = pd.DataFrame.from_dict(dict(Counter(np.argmax(result.phi[d], axis=1))), orient='index') / len(result.phi[d])
            phi_dist = phi_dist.merge(phi_d, how = 'outer', left_index=True, right_index=True)

        plt.figure()
        plt.boxplot(phi_dist)
        plt.show()
        
        
    # feature parameterization
    lambda_mu = pd.DataFrame(np.array(result.mk)).transpose()
    lambda_mu.index = result.vocabulary
    
    correlations_array = np.asarray(lambda_mu)
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
    
    sns.set(font_scale=0.5)
    sns.clustermap(lambda_mu, row_linkage=row_linkage, col_cluster=False, method="average")
    plt.savefig('%s_mu.png' % output)
    plt.close()


    # perplexity
    print 'Compute perplexity'
    logpw_d = []
    Nd_d = [] 
    
    test_genes = np.random.choice(result.test_genes, 100, replace=False)
    for g in test_genes[:2]:
        words_d = result.test_data[result.test_data.index == g]
        Nd_d.append(len(words_d))
        logpw_d.append(np.sum([np.log(np.sum([Theta.loc[g][k] * mvn_pdf(np.array(words_d.iloc[n]), result.mk[k], result.nuk[k] * result.Wk[k]) for k in xrange(result.K)])) for n in xrange(len(words_d))]))
    perplexity= np.exp(-np.sum(logpw_d) / np.sum(Nd_d))
    return perplexity


def mvn_pdf(x, mu, precision):
    # mu.shape: (D,)
    pdf =  np.power(np.linalg.det(precision), 0.5)/np.power(2*np.math.pi, len(mu)/2) *             np.exp(-0.5 * np.dot(np.dot((x-mu)[np.newaxis], precision), (x-mu)[np.newaxis].T))
    return pdf



ppp = []
kkk = []
datadir = ''

for k in range(4,30):
    inputfn = '%s/GeneLeast_150_K_%d.p' % (datadir, k)
    try:
        perplexity_K = plot_results(inputfn)
        ppp.append(perplexity_K)
        kkk.append(k)
        print k, perplexity_K
    except:
        continue


outfile = '%s/Perplexity_K'
np.savez(outfile, perplexity = ppp, clusterK = k)




