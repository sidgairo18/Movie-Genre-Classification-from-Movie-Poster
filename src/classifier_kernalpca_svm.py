from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm, preprocessing
import numpy as np


def change(k):
    
    n = k.shape[0]
    one_n = np.ones((n,n))/n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)
    
    return k


def reading():                                                          
                                                                        
    f = open('./data/arcene_train.data', 'r')                                                                   
    #f = open('./data2/madelon_train.data', 'r')                                                                   
    print "Reading data"                                                
    first = f.readline().split()
    first = list(map(int, first))                                       
                                                                        
    data = np.asarray(first)                                             
    
    for line in f:                                                      
        l = line.split()                                                
        l = list(map(int, l))                                           
        data = np.vstack((data, l))                           
                                                                        
    f.close()

    f = open('./data/arcene_train.labels')
    #f = open('./data2/madelon_train.labels')
    l1 = []
    print "Reading training labels"
    for line in f:
        l1.append(int(line))
    f.close()
    
    f = open('./data/arcene_valid.data', 'r')                                                                   
    #f = open('./data2/madelon_valid.data', 'r')                                                                   
    print "Reading validation data"

    first = f.readline().split()
    first = list(map(int, first))                                       
                                                                        
    data2 = np.asarray(first)                                             
    
    for line in f:                                                      
        l = line.split()                                                
        l = list(map(int, l))                                           
        data2 = np.vstack((data2, l))                           
                                                                        
    f.close()
    
    f = open('./data/arcene_valid.labels')
    #f = open('./data2/madelon_valid.labels')
    l2 = []
    print "Reading valid labels"
    for line in f:
        l2.append(int(line))
    f.close()

    print "Reading Complete"                                            
    return data, data2, l1, l2

def kernel_pca(data, gamma, n_components):
    
    sq_dists = pdist(data, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    k = exp(-gamma*mat_sq_dists)
    n = k.shape[0]
    one_n = np.ones((n,n))/n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)
    eigvals, eigvecs = eigh(k)

    x_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    return x_pc, k

def kernel_lda(data, l1, gamma):
   
   print "lda begins" 

   dump, k = kernel_pca(data, gamma, 10)

   k1 = []
   k2 = []

   for i in range(len(l1)):
       if l1[i] == -1:
           k1.append(k[i,:])
       else:
           k2.append(k[i,:])

   k1 = np.asarray(k1)
   k2 = np.asarray(k2)

   mean1 = np.mean(k1, axis=0)
   mean2 = np.mean(k2, axis=0)
   s1 = k1.shape[0]
   one1 = np.ones((s1,s1))/s1
   s2 = k2.shape[0]
   one2 = np.ones((s2,s2))/s2

   n1 = k1.T.dot((np.identity(s1)-one1).dot(k1))
   n2 = k2.T.dot((np.identity(s2)-one2).dot(k2))

   n = n1+n2
    
   a = np.linalg.inv(n).dot((mean2-mean1).T)
   print "lda ends"
   return a

if __name__ == "__main__":

    gamma = 1e-8

    data, data2, l1, l2 = reading()
    data_copy = np.copy(data)
    data3 = np.copy(data)
    
    asd = kernel_lda(data, l1, gamma)

    e, k = kernel_pca(data, gamma, 100)
    
    #svm part being done - testing

    clf = svm.SVC()
    new_data = k.dot(e)
    
    clf.fit(preprocessing.scale(new_data), l1)

    hits = 0

    k_new = []

    for i in range(data2.shape[0]):
        m = data_copy-data2[i,:]
        m = m.dot(m.T)
        m = m.diagonal()
        m = exp(-gamma*m)
        m = np.asarray(m)
        k_new.append(m)
        print i

    k_new = np.asarray(k_new)
    k2_new = np.copy(k_new)


    k_new = change(k_new)
    print k_new.shape

    k_new = k_new.dot(e)


    output = clf.predict(preprocessing.scale(k_new))
    
    hits = 0
    for i in range(len(l2)):
        if l2[i] == output[i]:
            hits += 1
    
    print hits
    
    asd = np.asarray(asd)

    clf2 = svm.SVC()
    new_data2 = k.dot(asd)

    new_data2 = np.matrix(new_data2)

    clf2.fit(preprocessing.scale(new_data2.T), l1)

    hits = 0

    k2_new = change(k2_new)
    k2_new = k2_new.dot(asd)
    k2_new = np.matrix(k2_new)


    output2 = clf2.predict(preprocessing.scale(k2_new.T))

    for i in range(len(l2)):
        if l2[i] == output2[i]:
            hits += 1

    print hits
