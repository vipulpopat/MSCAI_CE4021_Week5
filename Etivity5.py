%reset

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

        
class Log:
    DEBUG = 1
    INFO = 2
    ERROR = 3


class My_pca:
    """
    Perform the PCA on a dataset
    
    There is a lot of log statements in this class. I intend to remove
    them in the final code. Leaving them in place for the time being as they
    are useful for debugging. 
    
    BUG ALERT
    =========
    The eigen values calculated by this class match the ones calculated by 
    scikit. 
    
    However, it appears that one of the eigen vectors is the negative version 
    of the one calculted by scikit. Currently investigating the reason behind 
    this.
        
    """

    log_level = Log.ERROR
    nb_components = 2
    eigen_values = []
    eigen_vectors = []


    def __init__(self):
        """ init """


    def __log__(self, message, level=Log.INFO):
        """
        Log a message only if its log level is equal or superior to self.log_level
        """
        if level >= self.log_level:
            print(message)

        
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
        my_cov = np.cov(c_matrix, rowvar=False)        
        self.__log__("covariance:\n{}".format(my_cov), Log.DEBUG)
    
        # eigen values, eigen vectors
        eigen_values, eigen_vectors = eig(my_cov)
        self.__log__("eigen_values:\n{}".format(eigen_values))
        self.__log__("eigen_vectors:\n{}".format(eigen_vectors))     
        
        # order eigen values and eigen vectors       
        sorted_eigen_values_indexes = eigen_values.argsort()[::-1]
        sorted_eigen_values = eigen_values[sorted_eigen_values_indexes]
        sorted_eigen_vectors = eigen_vectors[sorted_eigen_values_indexes] 
        self.__log__("sorted_eigen_values_indexes:\n{}".format(sorted_eigen_values_indexes))
        self.__log__("sorted_eigen_values:\n{}".format(sorted_eigen_values))
        self.__log__("sorted_eigen_vectors:\n{}".format(sorted_eigen_vectors))

        # use nb_components to decide how many eigen vectors to keep
        filtered_sorted_eigen_values = sorted_eigen_values[:self.nb_components]
        filtered_sorted_eigen_vectors = sorted_eigen_vectors[:self.nb_components] 
        self.__log__("filtered_sorted_eigen_values:\n{}".format(sorted_eigen_values))
        self.__log__("filtered_sorted_eigen_vectors:\n{}".format(sorted_eigen_vectors))
        
        # calculate projection of dataset onto the eigen vector basis
        #P = eigen_vectors.T.dot(c_matrix.T)        
        #self.__log__("projected  :\n{}".format(P.T), Log.DEBUG)
        
        # save results as class variables
        self.eigen_values = filtered_sorted_eigen_values
        self.eigen_vectors = filtered_sorted_eigen_vectors
            
    
    def transform(self, data):
        """
        Calculate projection of dataset onto the eigen vector basis
        
        BUG:
        This method does work yet...
        """
        self.fit(data)
        self.projection = self.eigen_vectors.T.dot(data.T)
        
        self.__log__("eigen_values:\n{}".format(self.eigen_values), Log.DEBUG)
        self.__log__("eigen_vectors:\n{}".format(self.eigen_vectors), Log.DEBUG)
        self.__log__("projected  :\n{}".format(self.projection), Log.DEBUG)
        
        
        
def build_dataset():
    """
    Create a dataset
    """
    a_x = 0.05
    a_y= 10

    data =  np.matrix([[n*(1+a_x*(rand.random()-0.5)),4*n+ a_y*(rand.random()-0.5)] for n in range(20)])

    print(data)
    print(data.shape)
    return data


def scikit_pca( matrix, nb_components):
    """
    Calculate the PCA using Scikit APIs
    """
    pca = PCA(nb_components)
    pca.fit(matrix)

    return pca

      
def test():

    my_pca = My_pca()

    # Calculate PCA using scikit, nb_components=2
    pca = scikit_pca(data, 2)
    print()
    print("scikit.pca with nb_components=2")
    print("eigen_values:", pca.explained_variance_)
    print("eigen_vectors:", pca.components_)
    print("transformed_data:", pca.transform(data))

    # Calculate PCA using scikit, nb_components=1
    pca = scikit_pca(data, 1)
    print()
    print("scikit.pca with nb_components=1")
    print("eigen_values:", pca.explained_variance_)
    print("eigen_vectors:", pca.components_)
    print("transformed_data:", pca.transform(data))

    # Calculate PCA using homebrew code, nb_components=2
    my_pca.nb_components=2
    my_pca.fit(data)
    print()
    print("pca homebrew nb_components=", my_pca.nb_components)
    print("eigen_values:", my_pca.eigen_values)
    print("eigen_vectors:", my_pca.eigen_vectors)  
        
    # Calculate PCA using homebrew code, nb_components=1
    my_pca.nb_components=1
    my_pca.fit(data)
    print()
    print("pca homebrew nb_components=", my_pca.nb_components)
    print("eigen_values:", my_pca.eigen_values)
    print("eigen_vectors:", my_pca.eigen_vectors)
    
    # Calculate transformation using homebrew code, nb_coomponent=2
    my_pca.nb_components=2
    my_pca.transform(data)
    print()
    print("Transform using homebrew code, nb_components=", my_pca.nb_components)
    print("Transform:", my_pca.projection)
    
    # Calculate transformation using homebrew code, nb_coomponent=1
    my_pca.nb_components=1
    #my_pca.transform(data)
    #print()
    #print("Transform using homebrew code, nb_components=", my_pca.nb_components)
    #print("Transform:", my_pca.projection)
    

data = build_dataset()    
test()