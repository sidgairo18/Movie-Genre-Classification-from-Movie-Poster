from __future__ import division
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm, preprocessing
import numpy as np
import pdb

#labels - 0 for action, 1 for adventure, 2 for animation, 3 biography, 4 for comedy

features_dir = ['./features/FC7_Features_Action.txt','./features/FC7_Features_Adventure.txt', './features/FC7_Features_Animation.txt']#, './features/FC7_Features_Biography.txt', './features/FC7_Features_Comedy.txt']

def change(k):
    
    n = k.shape[0]
    one_n = np.ones((n,n))/n
    k = k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n)
    
    return k


def reading():                                                          
                                                                        
    f = open('./features/FC7_Features_Action.txt', 'r')

    print "Reading data"                 
    first = f.readline().split()
    first = list(map(float, first))                                       
                                                                        
    data = np.asarray(first)
    train_count = 100
    index = 1

    for i in range(len(features_dir)):

        print "Reading file", features_dir[i]
        
        for line in f:
            l = line.split()
            l = list(map(float, l))                                           
            data = np.vstack((data, l))
            index += 1
            if index >= train_count:
                break

        f.close()
        if (i+1) < len(features_dir):
            f = open(features_dir[i+1], 'r')
            index = 0

    print "Setting Labels"
    l1 = []
    l2 = []

    for i in range(len(features_dir)):
        for j in range(100):
            l1.append(i)
            l2.append(i)
    
    f = open('./features/FC7_Features_Action.txt', 'r')
    print "Reading validation data"
    
    index = 0
    for line in f:
        if index == train_count:
            first = line.split()
            break
        index += 1

    first = list(map(float, first))                                       
                                                                        
    data2 = np.asarray(first)
    test_count = 100
    index = train_count + 1

    for i in range(len(features_dir)):

        print "Reading file", features_dir[i]
        
        for line in f:
            l = line.split()
            l = list(map(float, l))
            if index >= train_count:
                data2 = np.vstack((data2, l))
            index += 1
            if index >= (train_count+test_count):
                break

        f.close()
        if (i+1) < len(features_dir):
            f = open(features_dir[i+1], 'r')
            index = 0
    
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
    
    #asd = kernel_lda(data, l1, gamma)

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
    print output
    
    hits = 0
    for i in range(len(l2)):
        if l2[i] == output[i]:
            hits += 1
    
    print hits

    exit()
    
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
