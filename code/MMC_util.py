# Multiplex Markov Chains

# Dane Taylor, Februaru, 2020


from pylab import *

import networkx as nx
import numpy as np
from scipy.linalg import block_diag
from scipy import sparse



### Basic Network Concepts

def transition_matrix(A): #creates a transition matrix from an adjacency matrix
	N = len(A)
	Dinv = zeros((N,N))
	for i in range(N): 
	    if sum(A[i])>0:
	        Dinv[i,i] = 1/sum(A[i])
	P = dot(Dinv,A)
	return P    

def google_matrix(A,alpha): #creates the PageRank, or Google, matrix
	P = transition_matrix(A)
	G = alpha*P + (1-alpha)/len(A) * ones(shape(A))    
	return G

def dom_left_eig(GG): #implements power method to compute dominant left eigenvector
	eigenValues,eigenVectors = eig(GG.T)        
	idx = eigenValues.argsort()[-1]   
	eigenValues = eigenValues[idx]
	x = np.abs(eigenVectors[:,idx])
	return x/sum(x)

def pagerank(A,alpha):  #computes pagerank centrality
	G = google_matrix(A,alpha)
	return dom_left_eig(G)



 ### Example Network Models

def star(N):# star graph in which node 1 is the hub
	A = np.zeros((N,N))
	A[0,1:] = np.ones(N-1)
	A += A.T
	return A

def undirected_chain(N):#chain is k-regular by adding self edges at endpoints
	A = np.zeros((N,N))
	for i in range(1,N):
	    A[i-1,i] = 1
	A += A.T
	A[0,0] = 1
	A[-1,-1] = 1    
	return A

def chain_Markov(I,a=0):
    if I==1:
        Pt = 1
    if I==2:
        Pt = array([[1-a,a],[a,1-a]])
    if I>2:
        Pt = diag((1-a)**(ones(I)))
        Pt += diag((a)/2**(ones(I-1)),-1)
        Pt += diag((a)/2**(ones(I-1)),1)
        Pt[0,0] += (a)/2
        Pt[-1,-1] += (a)/2
    
    return Pt

def get_chain_Pts(I,a,N):
    if type(a)==float or type(a)==int:
        Pts = [chain_Markov(I,a) for n in range(N)]
    if type(a) == np.ndarray:
        Pts = [chain_Markov(I,a[n]) for n in range(N)]
    return Pts

def get_random_Pts(I,a,N):
    Pts = []
    for n in range(N):
        aa = rand()*(1-a) + a
        #Pt = np.array([[1-aa,aa],[aa,1-aa]],dtype=float)
        Pt = chain_Markov(I,aa)
        Pts.append( Pt )
    return Pts

def get_increasing_Pts(I,a,N):
    Pts = []
    for n in range(N):
        aa = n*(1-a)/(N-1) + a
        #Pt = np.array([[1-aa,aa],[aa,1-aa]],dtype=float)
        Pt = chain_Markov(I,aa)
        Pts.append( Pt )
    return Pts

def get_decreasing_Pts(I,a,N):   
    Pts = get_increasing_Pts(I,a,N)
    Pts.reverse()
    return Pts




### Supracentrality

def build_block_diag(As):# returns diag(A^{(t)})
    As_block = block_diag(As[0],As[1]) 
    for n in range(2,len(As)):
        As_block = block_diag(As_block,As[n]) 
    return As_block

def build_sum_term(Pts,N):    
    if len(Pts) == N:
        T = len(Pts[0])
        X = zeros((N*T,N*T))
        for n in range(N):
            En = np.diag(np.arange(N)==n)*1
            X += kron(Pts[n],En)

        return X

    if len(Pts) == 1: return kron(Pts[0],eye(N))
    
    if len(Pts) != N:
        print('error! Wrong number of interlayer Markov chains.')
        return 0
    
    return 0

def supraCentralityMatrix(As,Pts,w,alpha):
    G = []
    for A in As:
          G.append(google_matrix(A,alpha)) #Determines Google matrices
    C = (1-w)*build_block_diag(G) + w*build_sum_term(Pts,len(As[0])) #Determines Supracentrality matirx

    return C

def supraCentrality(As,Pts,w,alpha):
    C = supraCentralityMatrix(As,Pts,w,alpha)
    joint = dom_left_eig(C)
    
    return joint.T.reshape(len(Pts[0]),len(As[0])).T #Reshapes eigenvalue




### Perturbation theory for small and large omega

def predicted_strong(As,Pts,alpha):
    
    #compute dominant eigenvectors of interlayer Markov chains 
    vs = np.zeros( (len(Pts),len(As)) )
    for n in range(len(Pts)):
        v = dom_left_eig(Pts[n])
        v /= sum(v)
        vs[n,:] = v
        
    #obtain an effective intralayer Markov chain
    X = zeros(shape(As[0]))
    for i in range(len(As)):
        X += np.dot( diag(vs[:,i]),google_matrix(As[i],alpha) )

    #use dominant eigenvector to get the weights
    weights = dom_left_eig(X)
    weights /= sum(weights)

    #use dominant eigenvector to obtain stationary distribution
    x2_strong = np.zeros(( len(Pts),len(As) ))
    for n in range(len(Pts)):
        x2_strong[n,:] = weights[n]* vs[n,:]

    return x2_strong,weights

def predicted_weak(As,Pts,alpha):
    
    #compute dominant eigenvectors of intralayer Markov chains 
    vs = np.zeros(( len(Pts),len(As) ))
    for i in range( len(As) ):
        v = dom_left_eig(google_matrix(As[i],alpha) )
        v /= sum(v)
        vs[:,i] = v
        
    #obtain an effective interlayer Markov chains 
    X_tilde = zeros( shape(Pts[0]) )
    for n in range(len(Pts)):
        X_tilde += np.dot(diag(vs[n,:]), Pts[n] )

    #use dominant eigenvector to get the weights
    weights = dom_left_eig(X_tilde)
    weights /= sum(weights)

    #use weights and dominant eigenvectors to obtain stationary distribution
    x2_weak = np.zeros(( len(Pts),len(As) ))
    for i in range(len(As)):
        x2_weak[:,i] = weights[i]* vs[:,i]

    return x2_weak,weights





### Study imbalance, convection and optimality

def study_imbalance(As,Pts,w,alpha):
    x2  = supraCentrality(As,Pts,w,alpha)    
    F =  np.dot(np.diag(x2.T.flatten()),supraCentralityMatrix(As,Pts,w,alpha))

    Delta = (F - F.T)/2
    total_imbalance = norm(Delta,'fro')

    pos_DF = np.maximum(F-F.T, np.zeros(np.shape(F)))

    return F,pos_DF,Delta,total_imbalance


def get_optimal_curves(ws,a_s,funs,As,alpha):
    N = len(As[0])
    I = len(As)

    imb_opts = zeros(( len(funs),len(a_s) ))
    conv_opts = zeros(( len(funs),len(a_s) ))
    conv_rates = np.zeros(( len(funs),len(a_s),len(ws) ))
    total_imbalances = np.zeros(( len(funs),len(a_s),len(ws) ))
    
    for tt,fun in enumerate(funs):
        for t,a in enumerate(a_s):
            Pts = fun(I,a,N) 
            
            for i,w in enumerate(ws):
                P = supraCentralityMatrix(As,Pts,w,alpha)
                conv_rates[tt,t,i] = - sort(-real(linalg.eig(P)[0]))[1]        
                _,_,_,total_imbalances[tt,t,i] = study_imbalance(As,Pts,w,alpha)
            
            imb_opts[tt,t] = ws[argmax(total_imbalances[tt][t])]
            conv_opts[tt,t] = ws[argmin(conv_rates[tt][t])]
        
    return conv_rates,total_imbalances,imb_opts,conv_opts


def compare_d_Delta(As,Pts,ws,alpha):
    I = len(As)
    N = len(Pts)
    dd =  array([sum(A,1) for A in As]).reshape(N*I)
        
    boo1 = []
    boo2 = []    
    tots = zeros(len(ws))
    Pearsons = zeros((len(ws),2))
    for t,w in enumerate(ws):
        #print(w)

        _,_,Delta,tots[t] = study_imbalance(As,Pts,w,alpha)
        
        AA = np.triu(build_block_diag(As))#intralayer edges
        row,col,weights = sparse.find(sparse.coo_matrix(AA!=0))
        delta = array([Delta[row[p],col[p]] for p in range(len(row))])
        d = array([ dd[row[p]] - dd[col[p]] for p in range(len(row))])
                
        AA = np.triu(build_sum_term(Pts,N))#interlayer edges
        row,col,weights = sparse.find(sparse.coo_matrix(AA!=0))
        delta2 = array([Delta[row[p],col[p]] for p in range(len(row))])
        d2 = array([ dd[row[p]] - dd[col[p]] for p in range(len(row))])
        
        Pearsons[t,0] = np.corrcoef(d,delta)[0,1]        
        Pearsons[t,1] = np.corrcoef(d2,delta2)[0,1]
        boo1.append(delta)
        boo2.append(delta2)
        
        print('Pearsons = '+ str(Pearsons[t,0]) + ', ' + str(Pearsons[t,1]))
        
    return Pearsons,tots,boo1,boo2

