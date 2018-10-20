

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

        
class Log:
    DEBUG = 1
    INFO = 2


class My_pca:
    """
    Perform the PCA on a dataset
            
    """

    log_level = Log.INFO
    nb_components = 2
   

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
        Fit the model with the matrix
        
        Steps:
        ======
        1) Find the mean of the dataset
        2) Center the dataset around the mean
        3) Calculate the covariance of the centered dataset
        4) Find the eigen values and eigen vectors using the covariance
        5) Order the eigen vectors based on their associated eigen values
        6) Only keep as many eigen vectors as defined in nb_components

        """
        self.matrix = matrix
        
        # Calculate mean values of each column from dataset
        m0 = np.mean(self.matrix[:,0])
        m1 = np.mean(self.matrix[:,1])
        
        # Center the columns by subtracting the corresponding mean
        c0 = matrix[:,0] - m0
        c1 = matrix[:,1] - m1
        
        # Create a centered matrix 
        self.c_matrix = []
        self.c_matrix = np.append(c0, c1, axis=1)
        
        # Calculate covariance of centered matrix
        my_cov = np.cov(self.c_matrix, rowvar=False)        
    
        # eigen values, eigen vectors
        eigen_values, eigen_vectors = eig(my_cov)
        
        # order eigen values and eigen vectors       
        sorted_eigen_values_indexes = eigen_values.argsort()[::-1]
        sorted_eigen_values = eigen_values[sorted_eigen_values_indexes]
        sorted_eigen_vectors = eigen_vectors.T[sorted_eigen_values_indexes]

        # use nb_components to decide how many eigen vectors to keep
        filtered_sorted_eigen_values = sorted_eigen_values[:self.nb_components]
        filtered_sorted_eigen_vectors = sorted_eigen_vectors[:self.nb_components] 
        
        # save results as class variables
        self.eigen_values = filtered_sorted_eigen_values
        self.eigen_vectors = filtered_sorted_eigen_vectors
        
        return(self.eigen_values, self.eigen_vectors)
         
    
    def transform(self, data):
        """
        Calculate projection of dataset onto the eigen vector basis
        """
        self.fit(data)
        self.__log__("="*80)
        self.__log__("PCA Homebrew nb_components={}\n".format(self.nb_components))
        
        self.__log__("eigen_values shape:{}".format(self.eigen_values.shape), Log.DEBUG)
        self.__log__("eigen_values:\n{}".format(self.eigen_values), Log.INFO)        

        self.__log__("eigen_vectors shape:\n{}".format(self.eigen_vectors.shape), Log.DEBUG)
        self.__log__("eigen_vectors:\n{}".format(self.eigen_vectors), Log.INFO)

        self.__log__("self.c_matrix shape:\n{}".format(self.c_matrix.shape), Log.DEBUG)
        self.__log__("self.c_matrix:\n{}".format(self.c_matrix), Log.DEBUG)
        
        self.projection = self.eigen_vectors.dot(self.c_matrix.T).T
        
        self.__log__("projected shape :\n{}".format(self.projection.shape), Log.DEBUG)
        self.__log__("projected  :\n{}".format(self.projection), Log.INFO)
        
        self.draw()
        return (self.eigen_values, self.eigen_vectors, self.projection)
        

    def draw(self):
        """
        Draws the projection of the centered dataset.
        """
        plt.plot(data[:,0], data[:,1], 'bo')
        if self.nb_components == 2:
            plt.plot(self.projection[:,0], self.projection[:,1], 'xr')
        elif self.nb_components == 1:
            plt.plot(self.projection, np.zeros_like(self.projection),'xr', label="Transformed dataset")            
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
    
    print("Eigen_values:", pca.explained_variance_)
    print("Eigen_vectors:", pca.components_)
    print("Transformed_data:", pca.transform(data))
        
    plt.plot(data[:,0], data[:,1], 'bo')

    if nb_components == 2:
        plt.plot(projection[:,0], projection[:,1], 'xr')
    elif nb_components == 1:
        plt.plot(projection, np.zeros_like(projection),'xr', label="MyPCA Transformed dataset")            
    plt.show()
     
    return pca

      
def test():

    my_pca = My_pca()

    # Calculate PCA using scikit, nb_components=2
    pca = scikit_pca(data, 2)

    # Calculate PCA using scikit, nb_components=1
    pca = scikit_pca(data, 1)
    
    # Calculate PCA using homebrew code, nb_components=2
    
    # Please note that the user can call the fit() API also. 
    # We are not doing it here because fit() is called by transform() internally
    # So this my_pca.fit(data) works but for brevity's sake I am simply 
    # calling my_pca.transform(data)
    my_pca.nb_components=2
    my_pca.transform(data)
        
    # Calculate PCA using homebrew code, nb_components=1
    my_pca.nb_components=1
    my_pca.transform(data)
    

data = build_dataset()    
test()