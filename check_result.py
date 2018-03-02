import sys
import pickle
import VI_LDA_GMM
from VI_LDA_GMM import LDA_GMM


import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import Counter
import seaborn as sns


from scipy.spatial import distance
from scipy.cluster import hierarchy



def plot_results(result):

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
    plt.savefig('%s_ELBO.png' % fn.replace('.p',''))
    plt.close()
    
    # Distribution of modules across genes
    Theta = np.array([x/np.sum(x) for x in result.gammma])
    row_linkage = hierarchy.linkage(distance.pdist(Theta), method='average')
    Theta = pd.DataFrame(Theta)
    Theta.index = result.genes

    sns.set(font_scale=0.8)
    sns.clustermap(Theta, row_linkage=row_linkage, col_cluster=False, method="average", cmap="YlGnBu")
    plt.savefig('%s_Theta.png' % fn.replace('.p',''))
    plt.close()

    # print 'Clustered genes based on Theta'
    # gene_clusters = pd.DataFrame(zip(hierarchy.fcluster(row_linkage, 1.15468), result.genes))
    # gene_clusters.columns = ['Cluster_ID', 'geneID']
    # for df_slice in gene_clusters.groupby(gene_clusters['Cluster_ID']):
    #     print np.array(df_slice[1]['geneID'])
        
        
    # feature parameterization
    lambda_mu = pd.DataFrame(np.array(result.mk)).transpose()
    lambda_mu.index = result.vocabulary
    
    correlations_array = np.asarray(lambda_mu)
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average')
    
    sns.set(font_scale=0.8)
    sns.clustermap(lambda_mu, row_linkage=row_linkage, col_cluster=False, method="average", cmap="YlGnBu")
    plt.savefig('%s_mu.png' % fn.replace('.p',''))
    plt.close()

    print 'Most important ten features for each module'
    import_fts = map(lambda t: np.array(lambda_mu.index)[np.argsort(t)[-10:]][::-1],np.transpose(np.array(lambda_mu)))
    for tp in xrange(result.K):
	print list(import_fts[tp])


    # histone modification, DNase, distance, contacting value
    interesting_fs = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'DNase_merged', 'tCD4', 'Distance']
    hms_color = plt.cm.rainbow(np.linspace(0, 1, len(interesting_fs)))
    x = [int(t) for t in range(result.K)]
    plt.figure()
    for tp in xrange(len(interesting_fs)):
	plt.scatter(x, np.array(lambda_mu.loc[interesting_fs[tp]]), color=hms_color[tp], label=interesting_fs[tp], edgecolors='none')

    plt.legend(loc='upper left')
    plt.savefig('%s_hms.png' % fn.replace('.p',''))
    plt.close()

    correlations_array = np.asarray(lambda_mu.loc[interesting_fs])
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method='average', metric='seuclidean')

    sns.set(font_scale=0.8)
    sns.clustermap(lambda_mu.loc[interesting_fs], row_linkage=row_linkage, col_cluster=False, method="average", metric='seuclidean', cmap="YlGnBu")
    plt.savefig('%s_hms_mu.png' % fn.replace('.p',''))
    plt.close()


    for k in xrange(result.K):
	cov_matrix_k = pd.DataFrame(result.nuk[k] * result.Wk[k])
	cov_matrix_k.index = result.vocabulary
	cov_matrix_k.columns = result.vocabulary
	cov_matrix_plot = cov_matrix_k.loc[interesting_fs]
	cov_matrix_plot = cov_matrix_plot[interesting_fs]

	sns.set(font_scale=0.8)
	sns.heatmap(cov_matrix_plot, cmap="YlGnBu")
	plt.savefig('%s_hms_cov_%d.png' % (fn.replace('.p',''), k))
	plt.close()


    if 1:
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




    # perplexity
    print 'Compute perplexity'

    logpw_d = []
    Nd_d = [] 
   
    try: 
	test_genes = np.random.choice(result.test_genes, 100, replace=False)
    except:
	test_genes = result.test_genes
    for g in test_genes:
        words_d = result.test_data[result.test_data.index == g]
        Nd_d.append(len(words_d))
        logpw_d.append(np.sum([np.log(1e-300+np.sum([Theta.loc[g][k] * mvn_pdf(np.array(words_d.iloc[n]), result.mk[k], result.nuk[k] * result.Wk[k]) for k in xrange(result.K)])) for n in xrange(len(words_d))]))
    perplexity= np.exp(-np.sum(logpw_d) / np.sum(Nd_d))

    return perplexity


def mvn_pdf(x, mu, precision):
    # mu.shape: (D,)
    pdf =  np.power(np.linalg.det(precision), 0.5)/np.power(2*np.math.pi, len(mu)/2) *             np.exp(-0.5 * np.dot(np.dot((x-mu)[np.newaxis], precision), (x-mu)[np.newaxis].T))
    return pdf




def main(output):
    with open(output, 'rb') as f:
        result = pickle.load(f)
    perplexity_K = plot_results(result)
    return perplexity_K



if __name__=='__main__':

    outdir = '/scratch/users/yhe23@jhu.edu/yuan/LDA_GMM/results'
    Moduel_K  = int(sys.argv[1])
    alphai = float(sys.argv[2])
    var_nu = float(sys.argv[3])
    var_lbd = float(sys.argv[4])
    episron = float(sys.argv[5])
    output = '%s/K_%d_varnu_%d_varlbd_%d_alphai_%s_episron_%s.p' % (outdir, Moduel_K, int(var_nu), int(var_lbd), str(alphai), str(episron))
    main(output)

