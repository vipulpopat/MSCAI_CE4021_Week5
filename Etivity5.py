#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:50:18 2018

@author: michel
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

# below are imports from the 
from numpy import array
from numpy import mean
from numpy import cov
        
class Log:
    DEBUG = 1
    INFO = 2

class My_pca:

    log_level = Log.INFO

    def __init__(self, nb_components):
        """ init """
        self.nb_components = nb_components;


    def __log__(self, message, level=Log.INFO):
        if level >= self.log_level:
            print(message)


    def scikit_pca(self, matrix):
        pca = PCA(n_components=2)
        pca.fit(matrix)
        self.__log__(("="*80))
        self.__log__("pca.explained_variance_:      \n{}".format(pca.explained_variance_))
        self.__log__("pca.components_:              \n{}".format(pca.components_))
        self.__log__("pca.explained_variance_ratio_:\n{}".format(pca.explained_variance_ratio_), Log.DEBUG)
        self.__log__("pca.singular_values_:         \n{}".format(pca.singular_values_), Log.DEBUG)
        self.__log__("pca.mean_:                    \n{}".format(pca.mean_), Log.DEBUG)
        self.__log__("pca.n_components_:            \n{}".format(pca.n_components_), Log.DEBUG)
        self.__log__("pca.noise_variance_:          \n{}".format(pca.noise_variance_), Log.DEBUG)

        return pca

        
    def fit(self, matrix):
        """ 
        Explicitly using array manipulation instead of 
        easier matrix operations 
        """
        self.matrix = matrix
        
        # Calculate mean values of each column from dataset
        m0 = np.mean(self.matrix[:,0])
        m1 = np.mean(self.matrix[:,1])
        self.__log__(("="*80))
        self.__log__("mean.col0:{} ".format(m0))
        self.__log__("mean.col1:{} ".format(m1))       
        
        # Center the columns by subtracting the corresponding mean
        c0 = matrix[:,0] - m0
        c1 = matrix[:,1] - m1
        self.__log__("c0       :{} ".format(c0), Log.DEBUG)
        self.__log__("c1       :{} ".format(c1), Log.DEBUG)       
        
        # Create a centered matrix 
        c_matrix = np.append(c0, c1, axis=1)
        self.__log__("centered_matrix:\n{}".format(c_matrix), Log.DEBUG)
        
        # Calculate covariance of centered matrix
        my_cov = np.cov(c_matrix.T)
        self.__log__("covariance:\n{}".format(my_cov), Log.DEBUG)
    
        # eigen values, eigen vectors
        values, vectors = eig(my_cov)
        self.__log__("eigenvalues:\n{}".format(values))
        self.__log__("eigenvectors:\n{}".format(vectors))     
        
        P = vectors.T.dot(c_matrix.T)        
        self.__log__("projected  :\n{}".format(P.T), Log.DEBUG)
        
        #TODO: order the eigen vectors using the eigenvalues
        #TODO: return a tuple
    
  
def build_dataset():
    a_x = 0.05
    a_y= 10

    data =  np.matrix([[n*(1+a_x*(rand.random()-0.5)),4*n+ a_y*(rand.random()-0.5)] for n in range(20)])

    print(data)
    print(data.shape)
    return data

      
def test():

    data = build_dataset()
    my_pca = My_pca(data)

    # Calculate PCA using scikit    
    my_pca.scikit_pca(data)

    # Calculate PCA using My_pca
    my_pca.fit(data)

test()

