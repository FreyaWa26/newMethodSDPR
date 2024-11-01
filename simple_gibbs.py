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
    

def sample_tau(state):
    shape = len(state['Y_'])/2+state['hyperparameters_']['a0k']
    rate = np.sum(np.array(state['residual'])*np.array(state['residual']))/2+state['hyperparameters_']['b0k']
    state['tau'] = np.random.gamma(shape, 1/rate, size=1)[0]
    #print('sample tau', state['tau'])

def vectorized_random_choice(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    k = np.clip(k, 0, len(items) - 1) 
    return k
    
def sample_assignment_beta(j, ld_boundaries,ref_ld_mat3, state, VS, rho):
    start_i = ld_boundaries[j][0]
    end_i = ld_boundaries[j][1]
    rho_1,rho_2,rho_3 = rho
    det_sigma = state['det_sigma_']
    tau = state['tau']
    N3 = state['N3_']
    num_snp = end_i - start_i
    num_var = state['num_clusters_']
    eta_sq = state['eta']**2
    B1 = state['B1'][j]; B2 = state['B2'][j];B3 = state['B3'][j]
    
    X1_i = state['X1_'][start_i:end_i]
    X2_i = state['X2_'][start_i:end_i]

    C=np.array(np.diag(state['C'][j]))
    ref_ld3 = ref_ld_mat3[j]
    
    residual = state['residual']
    
    b3 = state['eta']*np.dot(state['A3'][j], state['beta_margin3_'][start_i:end_i]) - state['eta']**2 * \
    (np.dot(B3, state['beta3'][start_i:end_i]) - np.diag(B3)*state['beta3'][start_i:end_i])
    
    state['b3'][start_i:end_i] = b3

    
    
    for i in range(num_snp):
        prob = np.zeros(num_var)
        beta1_i = state['beta1'][start_i+i]
        beta2_i = state['beta2'][start_i+i]
        #print('beta1',beta1_i,'beta2',beta2_i)
        b1 = tau*eta_sq *( C[i]*beta2_i+ B1[i,i]*beta1_i)+ \
            np.dot(X1_i[i],residual) *tau*state['eta']
        b2 = tau*eta_sq *( C[i]*beta1_i+ B2[i,i]*beta2_i)+ \
            np.dot(X2_i[i],residual) *tau*state['eta']
        # B should not change with different variance k
        B_i = np.array([b1,b2,N3*b3[i]])
        rando = False
        for k in range(1,num_var):
            var_k = state['var'][k]
            deno_k = det_sigma*var_k
            ak1 = tau*eta_sq*B1[i,i]+(1-rho_3**2)/deno_k
            ak2 = tau*eta_sq*B2[i,i]+(1-rho_2**2)/deno_k
            ak3 = N3 *eta_sq*B3[i,i]+(1-rho_1**2)/deno_k
            ck1 = (rho_3-rho_1*rho_2) / deno_k
            ck2 = (rho_2-rho_1*rho_3) / deno_k
            ck3 = (rho_1-rho_2*rho_3) / deno_k - tau*eta_sq*C[i]
            A_i = np.array([
                    [ak1, -ck3, -ck2],
                    [-ck3, ak2, -ck1],
                    [-ck2, -ck1, ak3]])
            A_inverse = np.array([
                    [ak2*ak3-ck1**2,  ck3*ak3+ck1*ck2, ck3*ck1+ak2*ck2],
                    [ck3*ak3+ck1*ck2, ak1*ak3-ck2**2 , ak1*ck1+ck2*ck3],
                    [ck3*ck1+ak2*ck2, ak1*ck1+ck2*ck3, ak1*ak2-ck3**2]
                    ])/np.linalg.det(A_i)
            
            exp_ele = 0.5 * B_i.T @ A_inverse @ B_i#np.dot(np.dot(B_i.T, np.linalg.inv(A_i)), B_i)
            non_exp = - 0.5 * np.log(np.linalg.det(A_i)) - 1.5*np.log(state['var'][k])  - .5*np.log(det_sigma) +np.log(state['p'][1]/state['p'][0])+ np.log( state['pi'][k]+1e-40) 
            prob[k] = exp_ele+non_exp
            if exp_ele ==float('inf'):
                rando = True
        if rando==True:
            state['assignment'][start_i+i]=0
            state['beta1'][start_i+i]=0
            state['beta2'][start_i+i]=0
            state['beta3'][start_i+i]=0
        else:
            prob = prob -np.mean(prob[1:])
            logexpsum = special.logsumexp(prob)
            prob_i = np.exp(prob -logexpsum)
            prob_i = prob_i/np.sum(prob_i)#adjust again
            k = np.random.choice(range(num_var), p=prob_i)
            #print('k',k)
            state['assignment'][start_i+i]= k
            if k==0:
                continue

            deno_k = det_sigma*state['var'][k]
            #print('sigmak',state['var'][k])
            ak1 = tau*eta_sq*B1[i,i]+(1-rho_3**2)/deno_k
            ak2 = tau*eta_sq*B2[i,i]+(1-rho_2**2)/deno_k
            ak3 = N3 *eta_sq*B3[i,i]+(1-rho_1**2)/deno_k
            ck1 = (rho_3-rho_1*rho_2) / deno_k
            ck2 = (rho_2-rho_1*rho_3) / deno_k
            ck3 = (rho_1-rho_2*rho_3) / deno_k - tau*eta_sq*C[i]
            A_i = np.array([
                    [ak1, -ck3, -ck2],
                    [-ck3, ak2, -ck1],
                    [-ck2, -ck1, ak3]])
            A_inverse = np.array([
                    [ak2*ak3-ck1**2,  ck3*ak3+ck1*ck2, ck3*ck1+ak2*ck2],
                    [ck3*ak3+ck1*ck2, ak1*ak3-ck2**2 , ak1*ck1+ck2*ck3],
                    [ck3*ck1+ak2*ck2, ak1*ck1+ck2*ck3, ak1*ak2-ck3**2]
                    ])/np.linalg.det(A_i)
            #print('B',B_i)
            #print('A',A_i)
            #print(A_inverse@B_i)
            beta_tmp= np.random.multivariate_normal(A_inverse @ B_i, A_inverse,size = 1)
            #print('beta_tmp',beta_tmp)
            b1_diff = beta_tmp[0,0] - state['beta1'][start_i+i]
            b2_diff = beta_tmp[0,1] - state['beta2'][start_i+i]
            #print('b1',b1_diff,'b2',b2_diff)
            state['beta1'][start_i+i]=beta_tmp[0,0]
            state['beta2'][start_i+i]=beta_tmp[0,1]
            state['beta3'][start_i+i]=beta_tmp[0,2]
            residual = residual- X1_i[i]*b1_diff - X2_i[i]*b2_diff

    state['residual']= residual 
    
def update_suffstats(state):
    suff_stats = dict(Counter(state['assignment']))
    suff_stats.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_']), list(suff_stats.keys())), 0))
    suff_stats = {k:suff_stats[k] for k in sorted(suff_stats)}
    state['suffstats'] = suff_stats
def sample_p(state):
    suffstats = np.array(list(state['suffstats'].values()))
    a = suffstats[0]
    b = np.sum(suffstats)-a
    print('a',a,'b',b)
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
        sample_assignment_beta(j,ld_boundaries, ref_ld_mat3, state, rho=rho, VS=True)
    update_suffstats(state)
    sample_V(state) 
    update_pi(state)
    sample_p(state)
    sample_tau(state)
    
    results = Parallel(n_jobs=-1)(
        delayed(parallel_task)(j, ld_boundaries, state, ref_ld_mat3)
        for j in range(len(ld_boundaries))
    )
    state['h2_1'] = np.var(np.dot(state['beta1'],state['X1_'])+np.dot(state['beta2'],state['X2_']))/np.var(state['Y_'])
    state['h2_3'] = sum(results)
