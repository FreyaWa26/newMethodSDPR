import numpy as np
import gzip, pickle
import scipy
import pandas as pd
import joblib
from scipy import linalg
import argparse
import sys
import statsmodels.api as sm
import gibbs,read

def SDPRX_m_gibbs(Y, X1,X2, beta_margin3,p,N3,  rho, ld_boundaries,  ref_ld_mat3, mcmc_samples, 
    burn,  tau ,save_mcmc = None, n_threads=4, VS=True): 
    l = len(beta_margin3)
    trace = { 'beta1':np.zeros(shape=(mcmc_samples, l )),
        'beta2':np.zeros(shape=(mcmc_samples, l)),
        'beta3':np.zeros(shape=(mcmc_samples, l)),
        'suffstats':[], 'h2_1':[], 'h2_3':[]}#'alpha':[], 'num_cluster':[],

    # initialize
    state = gibbs.initial_state(Y=Y, data1=X1, data2=X2,data3 = beta_margin3,p = p,N3=N3,ld_boundaries=ld_boundaries,tau = tau,k=100)
    state['residual'] = state['A']-np.dot(state['beta1'],state['X1_'])-np.dot(state['beta2'],state['X2_'])
    rho_1,rho_2,rho_3 = rho
    state['det_sigma_'] = 1-rho_1**2-rho_2**2-rho_3**2+2*rho_1*rho_2*rho_3
    gibbs.update_suffstats(state)#print(state['suffstats'])
    for j in range(len(ld_boundaries)):
        start_i = ld_boundaries[j][0]
        end_i = ld_boundaries[j][1]
        ref_ld3 = ref_ld_mat3[j]
        X1_i = state['X1_'][start_i:end_i]
        X2_i = state['X2_'][start_i:end_i]
        
        state['A3'].append( np.linalg.solve(ref_ld3+ state['N3_']*state['a']*np.identity(ref_ld3.shape[0]), ref_ld3))
        state['B3'].append( np.dot(ref_ld3, state['A3'][j]) )
        state['B1'].append(np.dot(X1_i, X1_i.T))
        state['B2'].append(np.dot(X2_i, X2_i.T))
        state['C'].append(np.dot(X1_i, X2_i.T))
        
    for i in range(mcmc_samples):
        # update everything
        gibbs.gibbs_stick_break(state, rho, ld_boundaries=ld_boundaries, ref_ld_mat3=ref_ld_mat3,
             n_threads=n_threads, VS=VS)
        # record the result
        trace['beta1'][i,] = state['beta1']*state['eta']
        trace['beta2'][i,] = state['beta2']*state['eta']
        trace['beta3'][i,] = state['beta3']*state['eta']
        
        if (i > burn):
            trace['h2_1'].append(state['h2_1'])
            trace['h2_3'].append(state['h2_3']*state['eta']**2)
        print(i,'h2_1: ', state['h2_1'], 'h2_3: ', state['h2_3']*state['eta']**2,' max beta1: ',max(trace['beta1'][i,]),' max beta2: ',max(trace['beta2'][i,]),'max beta3: ',max(trace['beta3'][i,]))
        
        # trace['pi'][i,:] = np.array(state['pi'].values())
        # trace['cluster_var'][i,:] = np.array(state['cluster_var'].values())
        # trace['alpha'].append(state['alpha'])
        # trace['num_cluster'].append( np.sum(np.array(state['pi'].values()) > .0001) )
        # trace['suffstats'].append(state['suffstats'])

        #util.progressBar(value=i+1, endvalue=mcmc_samples)

    # calculate posterior average
    poster_mean1 = np.mean(trace['beta1'][burn:mcmc_samples], axis=0)
    poster_mean2 = np.mean(trace['beta2'][burn:mcmc_samples], axis=0)
    poster_mean3 = np.mean(trace['beta3'][burn:mcmc_samples], axis=0)

    poster_se1 = np.std(trace['beta1'][burn:mcmc_samples], axis=0)
    poster_se2 = np.std(trace['beta2'][burn:mcmc_samples], axis=0)
    poster_se3 = np.std(trace['beta3'][burn:mcmc_samples], axis=0)

    
    print ('m_h2_1: ',np.median(trace['h2_1']),' m_h2_3: ' ,np.median(trace['h2_3']))
    #print('mean same assignment',np.mean(state['same_assig']))

    #print state['pi_pop']

    if save_mcmc is not None:
        df_beta1 = pd.DataFrame(trace['beta1'], columns=[f'beta1_{i}' for i in range(l)])
        df_beta2 = pd.DataFrame(trace['beta2'], columns=[f'beta2_{i}' for i in range(l)])
        df_beta3 = pd.DataFrame(trace['beta3'], columns=[f'beta3_{i}' for i in range(l)])
        df_beta1.to_csv('beta1.csv', index=False)
        df_beta2.to_csv('beta2.csv', index=False)
        df_beta3.to_csv('beta3.csv', index=False)

    return poster_mean1, poster_mean2, poster_mean3,poster_se1,poster_se2,poster_se3


def pipeline(args):
    
    # sanity check

    if args.bfile is not None and args.load_ld is not None:
        raise ValueError('Both --bfile and --load_ld flags were set. \
            Please use only one of them.')

    if args.bfile is None and args.load_ld is None:
        raise ValueError('Both --bfile and --load_ld flags were not set. \
            Please use one of them.')
    dat = read.Dat()
    read.get_size_vcf(args.phenopath, args.genopath, dat) 
    read.read_pheno(args.phenopath,dat)
    read.read_cov(args.covpath,dat)
    #read.read_lanc(args.genopath,args.msppath,dat) 
    print('Load summary statistics dataframe from {}'.format(args.sumpath))
    summ_stats =  pd.read_csv(args.sumpath,sep = "\t")   
    
    print('Load LD boundary from {}'.format(args.boundary))
    boundary_df = pd.read_csv(args.boundary, header=None)
    ld_boundaries  = boundary_df.to_numpy().astype(np.uintp)
    
    print('Load pre-computed reference LD from {}'.format(args.load_ld))
    ldmat_list = []
    with open(args.load_ld, 'rb') as f:
        while True:
                # Read rows and cols
            rows = f.read(8)
            if not rows:
                break
            cols = f.read(8)
            rows = int.from_bytes(rows, byteorder='little')
            cols = int.from_bytes(cols, byteorder='little')
                
                # Read matrix data
            data = f.read(rows * cols * 8)
            mat = np.frombuffer(data, dtype=np.float64).reshape((rows, cols))
            ldmat_list.append(mat)
    
    print('Load individual data from {}'.format(args.ss1))
    X1 = pd.read_csv(args.ss1,sep='\t',header=None)
    print('Load individual data from {}'.format(args.ss2))
    X2 = pd.read_csv(args.ss2,sep='\t', header=None)

    dat.n_snp = X1.shape[0]
    print('remain',dat.n_snp,'SNPs')
    beta3_hat = summ_stats['beta'] 
    
    rho1,rho2,rho3 = args.rho
    print('rho is ',rho1,rho2,rho3)
    print('Start MCMC ...')
    eta = 1
    rho = args.rho
    Y_all = dat.pheno ###note
    W = np.zeros((dat.n_ind, dat.n_cov))
    for i in range(dat.n_ind):
        for j in range(dat.n_cov):
            W[i, j] = dat.covar[j][i]
    
    model = sm.OLS(Y_all, W)
    results = model.fit()
    dat.pheno = Y_all-results.predict(W)
    after = pd.DataFrame(Y_all-results.predict(W));after.to_csv('after_cov_pheno.csv', index=False)
    print('Conditioned on the convariates.')
    eps = np.var(dat.pheno)
    tau = 1/eps
    
    beta_1,beta_2,beta_3,se1,se2,se3 = SDPRX_m_gibbs(dat.pheno,X1,X2, beta3_hat,args.p_sparse,args.N3, args.rho, ld_boundaries, ldmat_list, args.n_sim,args.burn,tau,args.out)
    
    print('Done!\nWrite output to {}'.format(args.out+'.txt'))
    
    results = pd.DataFrame({'beta1':beta_1,'beta2':beta_2,'beta3':beta_3,'se1':se1,'se2':se2,'se3':se3})
    summ_stats = pd.concat([summ_stats, results], axis=1)
    results.to_csv(args.out+'Results_'+args.ID+'.csv', index=False)
    summ_stats.to_csv(args.out+'Results_merged_'+args.ID+'.csv', index=False)



parser = argparse.ArgumentParser(prog='SDPRM',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Version 0.0.1 Test Only")

parser.add_argument('--phenopath', type=str, required=True,
                        help='Phenotype path. e.g. /home/tutor/height.txt')

parser.add_argument('--genopath', type=str, required=True,
                        help='Genotype path. e.g. /home/tutor/22.filter.vcf')
parser.add_argument('--covpath', type=str, required=True,
                        help='Covariate path. e.g. /home/tutor/covar.txt')

parser.add_argument('--ss1', type=str, required=True,
                        help='Path to individual statistics 1. e.g. /home/tutor/myss.txt')

parser.add_argument('--ss2', type=str, required=True,
                        help='Path to individual statistics 2. e.g. /home/tutor/myss.txt')

parser.add_argument('--sumpath', type=str, required=True,
                        help='msp path. e.g. /home/tutor/msp_path.txt')

parser.add_argument('--boundary', type=str, default=None,required=True,
                        help='LD boundary.')


parser.add_argument('--load_ld', type=str, default=None,required=True,
                        help='Prefix of the location to load calculated LD Reference file \
                        in pickled and gzipped format.')

parser.add_argument('--N3', type=int, default=None, required=True,
                        help='Number of individuals in summary statistic sile 3.')

parser.add_argument('--n_sim', type=int, default=None, required=True,
                        help='Number of simulations')

parser.add_argument('--rho', type=float, nargs=3, default=[0.8,0.5,0.3], 
                        help='Transethnic genetic correlation.')

parser.add_argument('--p_sparse', type=float, default=None,required=True,
                        help='Proportion of non-causal snps in simulation data.')

parser.add_argument('--pi_k', type=list, default=[1],
                        help='Distribution of sigma_k.')

parser.add_argument('--VS', type=bool, default=True, 
                        help='Whether to perform variable selection.')
parser.add_argument('--bfile', type=str, default=None,
                        help='Path to reference LD file. Prefix for plink .bed/.bim/.fam.')

parser.add_argument('--threads', type=int, default=2, 
                        help='Number of Threads used.')

parser.add_argument('--seed', type=int, default = 68,
                        help='Specify the seed for numpy random number generation.')

parser.add_argument('--burn', type=int, default=None,required=True,
                        help='Specify the total number of iterations to be discarded before \
                        Markov Chain approached the stationary distribution.')

parser.add_argument('--save_ld', type=str, default=None,
                        help='Prefix of the location to save calculated LD Reference file \
                        in pickled and gzipped format.')

parser.add_argument('--save_mcmc', type=str, default=None,
                        help='Prefix of the location to save intermediate output of MCMC \
                        in pickled and gzipped format.')

parser.add_argument('--out', type=str, required=True,
                        help='Prefix of the location for the output tab deliminated .txt file.')

parser.add_argument('--ID', type=str, required=True,
                        help='Profix of the name for the output .csv file.')




def main():
    if sys.version_info[0] != 3:
        print(sys.version_info[0],'ERROR: SDPR currently does not support Python 3')
        sys.exit(1)
    pipeline(parser.parse_args())

if __name__ == '__main__':
    main()



