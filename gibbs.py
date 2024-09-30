import numpy as np
from scipy import stats, special, linalg
from collections import Counter
from joblib import Parallel, delayed


def initial_state(Y, data1, data2,data3,p, N3, ld_boundaries,k, tau, alpha=1.0, a0k=.1, b0k=.1):
    num_clusters = k
    n_snp = len(data3)
    n_idv = len(Y)
    state = {
        'Y_':Y,
        'X1_': data1,
        'X2_': data2,
        'beta_margin3_': data3,
        'N3_':N3,
        'b1': np.zeros(n_snp),
        'b2': np.zeros(n_snp),
        'b3': np.zeros(n_snp),
        'B1': [],
        'B2': [],
        'B3': [],
        'A3':[],
        'C':[],
        'beta1': np.zeros(n_snp),
        'beta2': np.zeros(n_snp),
        'beta3': np.zeros(n_snp),
        'num_clusters_': num_clusters,
        'hyperparameters_': {
            "a0k": a0k,"b0k": b0k,
            "a0": 0.5,"b0": 0.5,
        },
        'suffstats': np.array([0]*(num_clusters)),
        'det_sigma_':0,
	    'assignment': np.zeros(n_snp),
        'pi': np.array([alpha / num_clusters]*num_clusters),
        'p': np.array([p,1-p]),
        'var': np.zeros(k),# possible variance
        'h2_1': 0,
        'h2_3': 0,
        'eta': 1,
	    'tau':tau,
        'V': np.zeros(num_clusters),
        'alpha':1,
        #'W1':np.zeros(n_idv),
        'residual':np.zeros(n_idv)
    }

    # define indexes
    state['a'] = 0.1/N3; state['c'] = 1
    state['A'] = Y#-state['alpha']*state['W1']*np.ones(len(Y))
    print('start assignment',sum(state['assignment']))
    
    return state   
    
   
def calc_b(j, state, ld_boundaries, ref_ld_mat3):
    
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    X1_i = state['X1_'][start_i:end_i]
    X2_i = state['X2_'][start_i:end_i]
    C=np.array(np.diag(state['C'][j]))
    ref_ld3 = ref_ld_mat3[j]
    B1 = np.diag(state['B1'][j]);B2 = np.diag(state['B2'][j]);B3 =state['B3'][j]
    residual = state['residual']
    b1 = np.dot(X1_i,residual) \
     + state['eta']*(C*np.array(state['beta2'][start_i:end_i])+ \
                     B1*np.array(state['beta1'][start_i:end_i]))
    b1 = b1*state['tau']*state['eta']
    
    b2 = np.dot(X2_i,residual) \
    + state['eta']*(C*np.array(state['beta1'][start_i:end_i])+\
                    B2*np.array(state['beta2'][start_i:end_i]))
    b2 = b2*state['tau']*state['eta']
    
    b3 = state['eta']*np.dot(state['A3'][j], state['beta_margin3_'][start_i:end_i]) - state['eta']**2 * \
    (np.dot(B3, state['beta3'][start_i:end_i]) - np.diag(B3)*state['beta3'][start_i:end_i])
    state['b1'][start_i:end_i] = b1
    state['b2'][start_i:end_i] = b2
    state['b3'][start_i:end_i] = b3

'''
def parallel_eta(j, ld_boundaries, state):
    tau = state['tau']
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    beta1 = state['beta1'][start_i:end_i]
    beta2 = state['beta2'][start_i:end_i]
    beta3 = state['beta3'][start_i:end_i]
    num = state['N3_']*np.dot(state['beta_margin3_'][start_i:end_i], np.dot(state['A3'][j], beta3))
    denum = state['N3_']*np.dot(beta3, np.dot(state['B3'][j], beta3))+\
        tau*(np.dot(beta1,np.dot(state['B1'][j],beta1))+np.dot(beta2,np.dot(state['B2'][j],beta2))+2*np.dot(beta1,np.dot(state['C'][j],beta2)))

    return num,denum


def sample_eta(state, ld_boundaries):

    results = Parallel(n_jobs=-1)(
        delayed(parallel_eta)(j, ld_boundaries, state)
        for j in range(len(ld_boundaries))
    )

    num_sum = state['tau']*np.dot(state['Y_'],np.dot(state['beta1'],state['X1_'])\
                                  +np.dot(state['beta2'],state['X2_']))\
    + sum(num for num, denum in results)
    denum_sum = sum(denum for num, denum in results)

     
    mu = num_sum / (denum_sum + 1e-6)
    var = 1.0 / (denum_sum+1e-6)#;print('var',var)
    state['eta']= np.sqrt(var)*stats.norm.rvs() + mu
    print('eta',state['eta'])  
'''
def sample_tau(state):
    shape = len(state['Y_'])/2+state['hyperparameters_']['a0k']
    rate = np.sum(np.array(state['residual'])*np.array(state['residual']))/2+state['hyperparameters_']['b0k']
    state['tau'] = np.random.gamma(shape, 1/rate, size=1)[0]
    print('sample tau', state['tau'])

def vectorized_random_choice(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    k = np.clip(k, 0, len(items) - 1) 
    return k
    
def sample_assignment(j, ld_boundaries,ref_ld_mat3, state, VS, rho):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    rho_1,rho_2,rho_3 = rho
    det_sigma = state['det_sigma_']
    tau = state['tau']
    N3 = state['N3_']
    num_snp = end_i - start_i
    num_var = state['num_clusters_']
    b1 = state['b1'][start_i:end_i]
    b2 = state['b2'][start_i:end_i]
    b3 = state['b3'][start_i:end_i]
    
    B1 = state['B1'][j]; B2 = state['B2'][j];B3 = state['B3'][j]
    C = state['C'][j]
    
    A = np.array([np.empty((3, 3)) for _ in range(num_snp)])
    B = np.empty((C.shape[0],3))
    log_prob = np.zeros((num_var,num_snp))
    eta_sq = state['eta']**2
    # shared with correlation
    for k in range(1,num_var):
        var_k = state['var'][k]
        deno_k = det_sigma*var_k
        ak1 = np.add.outer(.5*state['tau']*eta_sq*np.diag(B1), .5*(1-rho_3**2)/deno_k)
        ak2 = np.add.outer(.5*state['tau']*eta_sq*np.diag(B2), .5*(1-rho_2**2)/deno_k)
        ak3 = np.add.outer(.5*N3*eta_sq*np.diag(B3), .5*(1-rho_1**2)/deno_k)
        ck1 = (rho_3-rho_1*rho_2) / deno_k
        ck2 = (rho_2-rho_1*rho_3) / deno_k
        ck3 = np.subtract.outer((rho_1-rho_2*rho_3) / deno_k ,state['tau']*eta_sq*np.diag(C))
        for i in range(num_snp):
            A_i = np.array([
                    [2 * ak1[i], -ck3[i], -ck2],
                    [-ck3[i], 2 * ak2[i], -ck1],
                    [-ck2, -ck1, 2 * ak3[i]]
                    ])
            B_i = np.array([b1[i], b2[i], N3*b3[i]])
            exp_ele = 0.5 * np.dot(np.dot(B_i.T, np.linalg.inv(A_i)), B_i)
            non_exp = -0.5*np.log(np.linalg.det(A_i)) - 1.5*np.log(state['var'][k])  - .5*np.log(det_sigma) +np.log(state['p'][1]/state['p'][0])+ np.log( state['pi'][k]+1e-40)
            log_prob[k,i] = exp_ele + non_exp
    logexpsum = special.logsumexp(log_prob, axis=0,keepdims=True)
    prob_mat = np.exp(log_prob - logexpsum)
    #print('prob_mat',prob_mat.shape,'num_var',num_var)
    ##note
    state['assignment'][start_i:end_i] = vectorized_random_choice(prob_mat,range(num_var))
    #return vectorized_random_choice(prob_mat,range(num_var))
 
def sample_beta(j, state,  ld_boundaries,  ref_ld_mat3, rho, VS=True):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    X1 = state['X1_'][start_i:end_i];X2 = state['X2_'][start_i:end_i]
    A = state['residual']+np.dot(state['beta1'][start_i:end_i],X1)+np.dot(state['beta2'][start_i:end_i],X2)
    B3 = state['B3'][j];A3 = state['A3'][j]

    assignment = state['assignment'][start_i:end_i]

    num_snp = end_i - start_i
    
    beta1 = np.zeros(num_snp); beta2 =np.zeros(num_snp);beta3 = np.zeros(num_snp )

    # null
    # shared with correlation
    idx = assignment != 0#;print(sum(idx),len(assignment))
    nonzero = sum(idx)
    zeros = np.zeros((nonzero, nonzero))
    if sum(idx)==0 :
        # all SNPs in this block are non-causal
        print("non causal")
        pass
    else:
        var = [state['var'][int(i)] for i in assignment[idx]]
        # population LD matrix
        shrink_ld = np.block([[state['tau']*np.dot(X1[idx],X1[idx].T),     state['tau']*np.dot(X1[idx],X2[idx].T),           zeros],
                              [state['tau']*np.dot(X2[idx],X1[idx].T),      state['tau']*np.dot(X2[idx],X2[idx].T),      zeros],
                             [zeros,                             zeros,   state['N3_']*B3[idx,:][:,idx]]])
        #np.savetxt('compare.txt',compare,delimiter = ',')
        #print(shrink_ld)
        #print('B3',min(np.linalg.eigvalsh(B3[idx,:][:,idx])))
        diag = np.diag([1/x for x in var])
        #print('diag',diag)
        
        rho_1,rho_2,rho_3 = rho
        cov_matrix =  np.linalg.inv(np.array([[1, rho_1, rho_3],
                                [rho_1, 1, rho_2],
                                [rho_3, rho_2, 1]]))
        
        

        var_mat = np.zeros((3*nonzero,3*nonzero))
        for i in range(3):
            for k in range(3):
                var_mat[i*nonzero:(i+1)*nonzero, k*nonzero:(k+1)*nonzero] = cov_matrix[i, k] *diag
        
  
        mat = shrink_ld + var_mat + np.eye(var_mat.shape[0]) * 1e-10
        try:
            cov_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            print("Matrix is not invertible.")
        size = var_mat.size
             
        cov_mat = cov_mat/(state['eta']**2)
        if not np.allclose(cov_mat, cov_mat.T):
            cov_mat = (cov_mat + cov_mat.T) / 2
        cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * 1e-10
        ###note
        while min(np.linalg.eigvalsh(cov_mat))<0:
            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * max(1,-min(np.linalg.eigvalsh(cov_mat)))
        '''
        a=0.98
        cov_mat = a*cov_mat +(1-a)*np.eye(cov_mat.shape[0])
        '''
        if (min(np.linalg.eigvalsh(cov_mat))<0):
            print(j,'not definite')
        if not np.allclose(cov_mat, cov_mat.T):
            print(j,'not symmetric')
        # A matrix
        A_gamma = np.concatenate([state['tau']*np.dot(X1[idx],A),
                                  state['tau']*np.dot(X2[idx],A),
               state['N3_']*np.dot(A3[idx,:], state['beta_margin3_'][start_i:end_i])])
        A_gamma = A_gamma*state['eta']
        mu = np.dot(cov_mat, A_gamma)
        ##note
        #np.savetxt(f'MVNcov-{j}.csv', cov_mat, delimiter=',')
        #print(min(np.linalg.eigvalsh(cov_mat)),check_matrix_properties(cov_mat))
        beta_tmp = np.random.multivariate_normal(mu, cov_mat,size = 3)
        beta1[idx] = beta_tmp[1,0:nonzero]
        beta2[idx] = beta_tmp[1,nonzero:2*nonzero]
        beta3[idx] = beta_tmp[1,2*nonzero:3*nonzero]    
        state['beta1'][start_i:end_i] = beta1
        state['beta2'][start_i:end_i] = beta2
        state['beta3'][start_i:end_i] = beta3
        #A = state['residual']+np.dot(state['beta1'][start_i:end_i],X1)+np.dot(state['beta2'][start_i:end_i],X2)
        state['residual']=A - state['eta']*np.dot(beta1,X1) - state['eta']*np.dot(beta2,X2)
    

def update_suffstats(state):
    suff_stats = dict(Counter(state['assignment']))
    suff_stats.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_']), list(suff_stats.keys())), 0))
    suff_stats = {k:suff_stats[k] for k in sorted(suff_stats)}
    state['suffstats'] = suff_stats
def sample_p(state):
    suffstats = np.array(list(state['suffstats'].values()))
    a = suffstats[0]
    b = np.sum(suffstats)-a
    state['p'][0] = np.random.beta(a, b, 1)
    state['p'][1] = 1-state['p'][0]
    #print('sample p', state['p'][0])
    
def sample_sigma2(state, rho, VS=True):
    b = np.zeros(state['num_clusters_'])
    a = np.array(list(state['suffstats'].values() ))*1.5 + state['hyperparameters_']['a0k']
    table = [[] for i in range(state['num_clusters_'])]
    assignment = state['assignment']
    for i in range(len(assignment)):
        table[int(assignment[i])].append(i)
    rho_1,rho_2,rho_3 = rho
    det_sigma = state['det_sigma_']
    # shared with correlation
    for i in range(state['num_clusters_']):
        beta1 = state['beta1'][table[i]]
        beta2 = state['beta2'][table[i]]
        beta3 = state['beta3'][table[i]]
        b[i] = np.sum( ((1 - rho_3**2)*beta1**2 + (1 - rho_2**2)*beta2 **2 + (1 - rho_1**2)*beta3**2 \
         - 2*(rho_1 - rho_2*rho_3)*beta1*beta2  \
         - 2*(rho_2 - rho_1*rho_3)*beta1*beta3 \
         - 2*(rho_3 - rho_1*rho_2)*beta2*beta3 )/ 2*det_sigma ) + state['hyperparameters_']['b0k']
        
    out = np.array([0.0]*state['num_clusters_'])
    if VS is True:
        out[1:] = stats.invgamma(a=a[1:], scale=b[1:]).rvs()
        out[0] = 0
    else: 
        out = dict(zip(range(0, state['num_clusters_']), stats.invgamma(a=a, scale=b).rvs()))
    #print('a',max(a),np.argmax(a),'b',max(b),'var',max(out),np.argmax(out))
    state['var'] = out



def sample_V(state):
    suffstats = np.array(list(state['suffstats'].values()))
    a = 1 + suffstats[:-1]
    b = state['alpha'] + np.cumsum(suffstats[::-1])[:-1][::-1]
    sample_val = stats.beta(a=a, b=b).rvs()
    m = state['num_clusters_']
    
    if 1 in sample_val:
        idx = np.argmax(sample_val == 1)
        sample_val[idx+1:] = 0
        sample_return = dict(zip(range(1,m), sample_val))
        sample_return[m-1] = 0
    else:
        sample_return = dict(zip(range(1,m), sample_val))
        sample_return[m-1] = 1
    state['V'] = [0]+list(sample_return.values())
    
# Compute pi
def update_pi(state):
    #state['pi'][0] = state['pi_pop'][0]
    V = state['V']
    m = len(V)
    a = np.cumprod(1-np.array(V)[0:(m-1)])*V[1:]
    pi = dict()
    pi[0] = V[0]
    pi.update(dict(zip(range(1, m), a)))  
    # last p may be less than 0 due to rounding error
    if pi[m-1] < 0: 
        pi[m-1] = 0
    state['pi'] = list(pi.values())

def parallel_task(j, ld_boundaries, state, ref_ld_mat3):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    
    ref_ld3 = ref_ld_mat3[j]
    var3_contrib = np.sum(state['beta3'][start_i:end_i] * np.dot(ref_ld3, state['beta3'][start_i:end_i]))
    return var3_contrib
    
def gibbs_stick_break(state, rho, ld_boundaries, ref_ld_mat3,n_threads=4, VS=True):
    sample_sigma2(state,rho) 
    for j in range(len(ld_boundaries)):
        calc_b(j, state, ld_boundaries, ref_ld_mat3)
        sample_assignment(j,ld_boundaries, ref_ld_mat3, state, rho=rho, VS=True)
        sample_beta(j, state, ld_boundaries,ref_ld_mat3 , rho=rho, VS=True)
        #state['residual'] = state['A']-state['eta']*np.dot(state['beta1'],state['X1_'])-state['eta']*np.dot(state['beta2'],state['X2_']) 
        
    update_suffstats(state)
    sample_V(state) 
    update_pi(state)
    sample_p(state)
    sample_tau(state)
    '''
    sample_eta(state, ld_boundaries)##note
    state['assignment'] = np.concatenate(Parallel(n_jobs=n_threads, require='sharedmem')(delayed(sample_assignment)(j=j, ld_boundaries= ld_boundaries, ref_ld_mat3=ref_ld_mat3, state=state, rho=rho, VS=True) for j in range(len(ld_boundaries))))
    for j in range(len(ld_boundaries)):
        sample_beta(j, state, ld_boundaries,ref_ld_mat3 , rho=rho, VS=True)
    state['residual'] = state['A']-state['eta']*np.dot(state['beta1'],state['X1_'])-state['eta']*np.dot(state['beta2'],state['X2_']) 
    '''       
    results = Parallel(n_jobs=-1)(
        delayed(parallel_task)(j, ld_boundaries, state, ref_ld_mat3)
        for j in range(len(ld_boundaries))
    )
    state['h2_1'] = np.var(np.dot(state['beta1'],state['X1_'])+np.dot(state['beta2'],state['X2_']))/np.var(state['Y_'])
    state['h2_3'] = sum(results)
