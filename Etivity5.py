

import numpy as np
import random as rand
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA


class Log:
    DEBUG = 1
    INFO = 2


class Algo:
    SCIKIT = 1
    HOMEBREW = 2


class My_pca:
    """
    Perform the PCA on a dataset

    This class provides PCA method for calculating the fit(), transform() 
    and inverse() on a dataset.

    This class is aimed at comparing 2 types of approaches:
        1) The Scikit approach which is leveraging the SVD algorithm
        2) The Course's approach which is leveraging the covariances and
           eigen values. In the rest of the code, I refer to the
           "Course approach" as "homebrew".

    Users of this class can specify these parameters:
        1) algo: The algorithm to use: SCIKIT or HOMEBREW (default)
        2) nb_components: The number of dimensions to use (default 2)
        3) show_graph: If set to True (default), plot the dataset and
                       transformation.
        4) log_level: Used during debugging. (default is INFO level)

    """

    algo = Algo.HOMEBREW
    nb_components = 2

    log_level = Log.INFO
    show_graph = True


    def __init__(self):
        """ init """


    def __log__(self, message, level=Log.INFO):
        """
        Log a message only if its log level is equal or superior to
        self.log_level
        """
        if level >= self.log_level:
            print(message)


    def fit(self, matrix):
        """
        Defer the calculation of the fit to the desired algorithm
        """
        if self.algo == Algo.HOMEBREW:
            return self.fit_homebrew_(matrix)
        else:
            return self.fit_scikit_(matrix)


    def fit_scikit_(self, matrix):
        """
        Fit the model with the matrix using the Scikit APIs
        """
        pca = PCA(self.nb_components)
        pca.fit(matrix)
        self.eigen_values = pca.explained_variance_
        self.eigen_vectors = pca.components_
        return (self.eigen_values, self.eigen_vectors)


    def fit_homebrew_(self, matrix):
        """
        Fit the model with the matrix

        Steps:
        ======
        1) Find the mean of the dataset
        2) Center the dataset around the mean of each column
        3) Calculate the covariance of the centered dataset
        4) Find the eigen values and eigen vectors using the covariance
        5) Order the eigen vectors based on their associated eigen values
        6) Only keep as many eigen vectors as defined in nb_components

        """
        self.matrix = matrix

        # Calculate mean values of each column from dataset
        self.m = np.mean(self.matrix.T, axis=1)

        # Center the columns by subtracting the corresponding mean
        self.c_matrix = []
        self.c_matrix = self.matrix - self.m.T

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
        Defer the calculation of the transformation to the desired algorithm
        """
        if self.algo == Algo.HOMEBREW:
            return self.transform_homebrew_(data)
        else:
            return self.transform_scikit_(data)


    def transform_scikit_(self, data):
        """
        Calculate the projection of the dataset onto the eigen vector basis
        using the Scikit APIs
        """
        self.__log__("="*80)
        self.__log__("PCA Scikit nb_components={}\n".format(self.nb_components))

        self.matrix = data 
        pca = PCA(self.nb_components)
        pca.fit(data)
        self.eigen_values = pca.explained_variance_
        self.eigen_vectors = pca.components_
        self.projection = pca.transform(data)

        self.__log__("eigen_values:\n{}".format(self.eigen_values), Log.INFO)
        self.__log__("eigen_vectors:\n{}".format(self.eigen_vectors), Log.INFO)
        self.__log__("projected  :\n{}".format(self.projection), Log.INFO)

        self.draw()
        return (self.eigen_values, self.eigen_vectors, self.projection)


    def transform_homebrew_(self, data):
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


    def inverse(self, data):
        """
        Defer the calculation of the inverse to the desired algorithm
        """
        if self.algo == Algo.HOMEBREW:
            return self.inverse_homebrew_(data)
        else:
            return self.inverse_scikit_(data)


    def inverse_scikit_(self, data):
        """
        Compute the inverse of the transform using Scikit
        """
        pca = PCA(self.nb_components)
        pca.fit(data)
        data_pca = pca.transform(data)
        return pca.inverse_transform(data_pca)


    def inverse_homebrew_(self, data):
        """
        Compute the inverse of the transform using Homebrew
        """
        reduced = self.eigen_vectors.T.dot(data[2].T).T 

        reduced_uncentered = reduced + self.m.T

        return reduced_uncentered

        
    def draw(self):
        """
        Draws the projection.
        """
        if self.show_graph == False:
            return

        plt.title("Dataset compared \n with it's transformation")
        plt.plot(data[:, 0], data[:, 1], 'bo', label="Original Data")
        if self.nb_components == 2:
            plt.plot(self.projection[:, 0], self.projection[:, 1], 'xr', label="Transformed dataset. \n(2 eigen vectors)")
        elif self.nb_components == 1:
            plt.plot(self.projection, np.zeros_like(self.projection), 'xr', label="Transformed dataset. \n(1 eigen vector)")

        plt.legend(loc='best')
        plt.show()


def build_dataset():
    """
    Create a dataset
    """
    a_x = 0.05
    a_y = 10

    data = np.array ([[n * (1 + a_x * (rand.random()-0.5)), 4 * n + a_y * (rand.random() - 0.5)] for n in range(20)])

    print("Data:\n", data)
    print(data.shape)
    return data


def test():
    """
    Using Sckikit, calculate the transform of the dataset
    Using the algorithm presented in the course, calculate the fit and transform for the dataset
    """

    # Test with scikit
    my_scikit = My_pca()
    my_scikit.algo = Algo.SCIKIT

    my_scikit.nb_components = 2
    my_scikit.transform(data)

    my_scikit.nb_components = 1
    my_scikit.transform(data)

    # Test with homebrew
    my_homebrew = My_pca()
    my_homebrew.algo = Algo.HOMEBREW

    my_homebrew.nb_components = 2
    my_homebrew.transform(data)

    my_homebrew.nb_components = 1
    my_homebrew.transform(data)


def test_sckikit_nb_components_1_and_2():
    """
    Use the Scikitlean's PCA class with n_components=2 and n_components=1
    and observe the differences. In the cell directly below, comment on
    what you have observed.
    """
    my_scikit = My_pca()
    my_scikit.algo = Algo.SCIKIT

    my_scikit.nb_components = 2
    my_scikit.transform(data)

    my_scikit.nb_components = 1
    my_scikit.transform(data)


def reflection_1():
    """
    For the case where n_components = 1, compare the resulting dataset of your
    transform method with the resulting dataset from Scikitlearn’s transform
    method by plotting the points on an XY plot.
    If there are any differences, explain these in a comment directly under
    the cell with your plot.
    """

    print("#"*80)
    print("Reflection Question 1")
    print("#"*80)

    my_scikit = My_pca()
    my_scikit.algo = Algo.SCIKIT
    my_scikit.nb_components = 1
    my_scikit.show_graph = False
    my_scikit.transform(data)

    my_homebrew = My_pca()
    my_homebrew.algo = Algo.HOMEBREW
    my_homebrew.nb_components = 1
    my_homebrew.show_graph = False
    my_homebrew.transform(data)

    plt.title("Comparing Scikit and Homebrew projections")
    plt.plot(my_scikit.projection, np.zeros_like(my_scikit.projection), 'or', label="Scikit transformed dataset")
    plt.plot(my_homebrew.projection, np.zeros_like(my_homebrew.projection), 'xb', label="Homebrew transformed dataset")
    plt.legend(loc='best')
    plt.show()


def reflection_2():
    """
    For the case where n_components = 1, compare the dataset resulting from
    your transform method with the original dataset by plotting the points
    on an XY plot.
    Comment on the differences between original and transformed data in the
    cell directly below your plot. In your comment, explain why and how PCA
    can be used for dimensionality reduction
    
    NOTE:
    =====    
    In the notes there is a mention of the inverse transform.
    Here we are using the inverse transform to figure out how close the
    inverse will be from the original dataset.

    """
    print("#"*80)
    print("Reflection Question 2")
    print("#"*80)

    my_homebrew = My_pca()
    my_homebrew.algo = Algo.HOMEBREW
    my_homebrew.nb_components = 1
    transformed_data = my_homebrew.transform(data)
    
    reduced_data = my_homebrew.inverse(transformed_data)
    
    plt.title("Dataset compared \n with its reduced form")
    plt.plot(data[:,0], data[:,1], 'or', label='Original data') 
    plt.plot(reduced_data[:,0], reduced_data[:,1],'xg', label='Reduced data') 
    plt.legend(loc='best')
    plt.show()
    
       
    
data = build_dataset()

test()
test_sckikit_nb_components_1_and_2()
reflection_1()
reflection_2()
