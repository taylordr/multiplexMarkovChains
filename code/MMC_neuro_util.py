# Multiplex Markov Chains
#
# this script contains helper functions for studying 
# the MMC representation of the brain-activity data
#
# Dane Taylor, June 30, 2020


from pylab import *

import numpy as np
from scipy.linalg import block_diag
from scipy import sparse
import scipy.io
import matplotlib.colors as colors

# these are the MMC scrips
from MMC_util import *
from MMC_plot_util import *


# compute degrees
def get_degrees(Peoples_As):
    I = len(Peoples_As[0])
    Degrees = zeros(( len(Peoples_As),len(Peoples_As[0][0]),I))
    for person in range(len(Peoples_As)):
        for i in range(I):
            Degrees[person,:,i] = sum(Peoples_As[person][i],0)
    return Degrees

# permute nodes so their order is given by the passed variable new_ids
def permute_As(As,new_ids):
    Bs = []    
    for i in range(len(As)):
        temp = As[i][new_ids]
        temp = temp.T
        temp = temp[new_ids]
        Bs.append(temp.T)
    return Bs


## Loading the data
def load_mat_file(data_load_specs):
    mat = scipy.io.loadmat(data_load_specs['file_name'])
    
    # healthy persons data
    hc_data = mat['HC'][0]
    hc_matrices = [hc_data[person][3][0] for person in range(len(hc_data))]

    # Alzheimers persons data
    ad_data = mat['AD'][0]
    ad_matrices = [ad_data[person][3][0] for person in range(len(ad_data))]

    
    I = len(ad_matrices[0]) # 7 frequency bands
    N = len(ad_matrices[0][0]) # 148 brain regions    
    
    Peoples_As = hc_matrices + ad_matrices # combine the lists
    
    
    # after loading data, potentially preprocess it before analysis

    if data_load_specs['self_edges'] == False:
        for p in range(len(Peoples_As)):
            for i in range(I):
                for n in range(N):
                    Peoples_As[p][i][n,n] = 0 # make self-edges 0
              
    if data_load_specs['unweighted'] == True:
        for person,As in enumerate(Peoples_As):
            for i in range(len(As)):   
                Peoples_As[person][i] = np.array(Peoples_As[person][i] >= data_load_specs['weight_threshold'],dtype=int)
                
    if data_load_specs['degree_sort'] == True:        
        Degrees = get_degrees(Peoples_As)# intralayer degree for every [person,node,layer]
        dd = mean(Degrees,0)# mean intralayer degrees across all people
        new_ids = argsort(-np.sum(dd,1))# sort from large to small the nodes' total degrees, averaged across people
        
        for t,As in enumerate(Peoples_As):
            Peoples_As[t] = permute_As(As,new_ids)
    
    return Peoples_As,N,I


# Visualize some layers for a person
def plot_As(Peoples_As,ids):
    I = len(Peoples_As[0])

    f1 = plt.figure(figsize=(14,6));
    for t,person in enumerate(ids):
        for i in range(I):
            plt.subplot(len(ids),I,I*t+i+1)
            plt.imshow(Peoples_As[person][i],cmap='hot')            
            if i==0:
                plt.ylabel('node, $n$')
            if t==(len(ids)-1):
                plt.xlabel('node, $n$')
    plt.tight_layout()
    return  


## Study the intralayer degrees
def plot_degrees(Degrees):
    num_people,N,I = shape(Degrees)
    f, ax = plt.subplots(2,2,figsize=(6,5),sharey='col')
    ranges = [range(25),range(25,50)]
    titles = ['Healthy People','Alzheimers People']
    for t in range(2): 
        dd = mean(Degrees[ranges[t]],0)
        ax[t,0].plot(dd,linewidth=.5,alpha=.8)
        ax[t,0].plot(mean(dd,1),'k',linewidth=2)
        ax[t,1].set_xlim([0,6])
        
        ax[t,1].plot(sum(Degrees[ranges[t]],axis=1).T,'k',alpha=.1);
        ax[t,1].plot(sum(dd,0),'k',linewidth=2)
        ax[t,0].set_xlim([0,N])
        ax[t,0].set_ylabel('$\d_n^{(i)}$')
        ax[t,1].set_ylabel('$\sum_n d_n^{(i)}$')        
        
    ax[0,0].set_ylabel('Healthy Persons \n\n $d_n^{(i)}$')
    ax[1,0].set_ylabel('Alzheimers Persons \n\n $d_n^{(i)}$')

    ax[0,0].set_title('Intralayer Degrees')
    ax[0,1].set_title('Edges per Layer')
    ax[1,0].set_xlabel('node, $n$')
    ax[1,1].set_xlabel('layer, $i$')
    ax[0,0].legend([str(i) for i in range(I)])
    
    plt.tight_layout()
    return
