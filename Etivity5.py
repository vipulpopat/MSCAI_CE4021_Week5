import numpy as np
import random as rand
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

        
class Log:
    DEBUG = 1
    INFO = 2
    ALWAYS_SHOW = 3


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

    log_level = Log.INFO
    nb_components = 2
    eigen_values = []
    eigen_vectors = []
    c_matrix = []


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
        self.__log__("mean.col0:{} ".format(m0))
        self.__log__("mean.col1:{} ".format(m1))       
        
        # Center the columns by subtracting the corresponding mean
        c0 = matrix[:,0] - m0
        c1 = matrix[:,1] - m1
        self.__log__("c0       :{} ".format(c0), Log.DEBUG)
        self.__log__("c1       :{} ".format(c1), Log.DEBUG)       
        
        # Create a centered matrix 
        self.c_matrix = []
        self.c_matrix = np.append(c0, c1, axis=1)
        self.__log__("centered_matrix:\n{}".format(self.c_matrix), Log.DEBUG)
        
        # Calculate covariance of centered matrix
        my_cov = np.cov(self.c_matrix, rowvar=False)        
        self.__log__("covariance:\n{}".format(my_cov), Log.DEBUG)
    
        # eigen values, eigen vectors
        eigen_values, eigen_vectors = eig(my_cov)
        self.__log__("eigen_values:\n{}".format(eigen_values), Log.INFO)
        self.__log__("eigen_vectors:\n{}".format(eigen_vectors), Log.INFO)     
        
        # order eigen values and eigen vectors       
        sorted_eigen_values_indexes = eigen_values.argsort()[::-1]
        sorted_eigen_values = eigen_values[sorted_eigen_values_indexes]
        sorted_eigen_vectors = eigen_vectors[sorted_eigen_values_indexes] 
        self.__log__("sorted_eigen_values_indexes:\n{}".format(sorted_eigen_values_indexes), Log.DEBUG)
        self.__log__("sorted_eigen_values:\n{}".format(sorted_eigen_values), Log.DEBUG)
        self.__log__("sorted_eigen_vectors:\n{}".format(sorted_eigen_vectors), Log.DEBUG)

        # use nb_components to decide how many eigen vectors to keep
        filtered_sorted_eigen_values = sorted_eigen_values[:self.nb_components]
        filtered_sorted_eigen_vectors = sorted_eigen_vectors[:self.nb_components] 
        self.__log__("filtered_sorted_eigen_values:\n{}".format(filtered_sorted_eigen_values), Log.INFO)
        self.__log__("filtered_sorted_eigen_vectors:\n{}".format(filtered_sorted_eigen_vectors), Log.INFO)
        
        # save results as class variables
        self.eigen_values = filtered_sorted_eigen_values
        self.eigen_vectors = filtered_sorted_eigen_vectors
            
    
    def transform(self, data):
        """
        Calculate projection of dataset onto the eigen vector basis
        """
        self.fit(data)
        
        self.__log__("eigen_values shape:{}".format(self.eigen_values.shape), Log.DEBUG)
        self.__log__("eigen_values:\n{}".format(self.eigen_values), Log.DEBUG)        
        self.__log__("eigen_vectors shape:\n{}".format(self.eigen_vectors.shape), Log.DEBUG)
        self.__log__("eigen_vectors:\n{}".format(self.eigen_vectors), Log.DEBUG)
        
        self.projection = self.eigen_vectors.T.dot(self.c_matrix.T).T
        
        self.__log__("projected shape :\n{}".format(self.projection.shape), Log.INFO)
        self.__log__("projected  :\n{}".format(self.projection), Log.INFO)
        
        plt.plot(data[:,0], data[:,1], 'bo')
        plt.plot(self.projection[:,0], self.projection[:,1], 'xr')
        plt.show()
        

def build_dataset():
    """
    Create a dataset
    """
    a_x = 0.05
    a_y= 10

    data =  np.matrix([[n*(1+a_x*(rand.random()-0.5)),4*n+ a_y*(rand.random()-0.5)] for n in range(20)])

    print("Data:\n", data)
    print(data.shape)
    return data


def scikit_pca( matrix, nb_components):
    """
    Calculate the PCA using Scikit APIs
    """
    print("="*80)
    print("Scikit.pca with nb_components=",nb_components)

    pca = PCA(nb_components)
    pca.fit(matrix)
    projection = pca.transform(matrix)
    
    if nb_components == 2:
        plt.plot(data[:,0], data[:,1], 'bo')
        plt.plot(projection[:,0], projection[:,1], 'xr')
        plt.show()

    print("Eigen_values:", pca.explained_variance_)
    print("Eigen_vectors:", pca.components_)
    print("Transformed_data:", pca.transform(data))
        
    return pca

      
def test():

    my_pca = My_pca()

    # Calculate PCA using scikit, nb_components=2
    pca = scikit_pca(data, 2)

    # Calculate PCA using scikit, nb_components=1
    pca = scikit_pca(data, 1)
    
    # Calculate PCA using homebrew code, nb_components=2
    my_pca.nb_components=2
    print("="*80)
    print("PCA Homebrew nb_components=", my_pca.nb_components, "\n")
    #my_pca.fit(data)
    my_pca.transform(data)
        
    # Calculate PCA using homebrew code, nb_components=1
    my_pca.nb_components=1
    print("="*80)
    print("PCA Homebrew nb_components=", my_pca.nb_components, "\n")
    my_pca.fit(data)
    

data = build_dataset()    
test()