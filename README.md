# Week5

Purpose

The purpose of this E-tivity is to help you become confident in using Numpy’s functionality for matrix manipulation and to learn about a very useful data processing technique: principal components analysis, or PCA. You will find that using Numpy is much easier for matrix manipulation than what you have done to date!

Task (Complete by Saturday Week 5)

Making use of Numpy, write a Python class to apply the PCA transform to the provided (see Notebook) data set. Compare the output of your implementation to the PCA functionality provided by the Scikitlearn module.

    Create a 'fit' method that calculates the eigen vectors and eigen values of your dataset. Compare your results to the output of Scikitlearn's fit method and document your findings as a comment (use markdown) directly under the cell with your PCA class.
    Use the Scikitlean's PCA class with n_components=2 and n_components=1 and observe the differences. In the cell directly below, comment on what you have observed.
    Add a property to your class and initialise this property in a suitable fashion to allow you to choose the number of principal components similar to the Scikitlearn PCA class.
    Store those results from your fit method that are required to transform the data set, in suitable class properties.
    Create a 'transform' method to perform the PCA data transformation on your data set using the parameters obtained using your 'fit' method.

 

N.B.:

    Limit your code to the aspects explicitly listed. 
    Use the Jupyter Notebook provided in the repository for week 5. This notebook contains the data that needs to be transformed.
    The required modules have already been imported for you. You should not import any other modules.
    If you find creating a class with this functionality daunting, please start by creating normal functions in your notebook. If time permits, you can then change to use of a class later. 

 

HINTS:

    Numpy.mean() will 'flatten' your tensor by default. To obtain the mean along a given axis, you may use the axis parameter.

    Numpy.cov() assumes by default that data is presented with one observation per column. This can be changed using the rowvar parameter. 

    A Numpy.matrix is a convenient way of performing the matrix operations required for PCA whilst retaining a matrix/vector like structure. Use of this class is discouraged, but would form a good starting point for tackling this week's challenge. Once you have the code working with the matrix class, changing to arrays is relatively straight forward.

    You can use Scikitlearn as follows to check the Eigen vectors that you have found with your 'fit' mehod:

pca = PCA(n_components=2)
pca.fit(data)
print(pca.components_)

    You can use Scikitlearn to obtain 

GIT push your implementation and post your manual calculations to E-tivity 5: Linear Algebra in Numpy and Beyond and provide the name of your branch.

Respond (Complete by Wednesday Week 6)

Respond to a post of one of your peers with a respectful and in-depth assessment of the implementation with a view to pointing out potential improvements or sound alternative solutions.

Reflect (Complete by Saturday Week 6)

With your code (containing any corrections you have made based on your peers’ feedback), do the following:

    For the case where n_components = 1, compare the resulting dataset of your transform method with the resulting dataset from Scikitlearn’s transform method by plotting the points on an XY plot. If there are any differences, explain these in a comment directly under the cell with your plot.
    For the case where n_components = 1, compare the dataset resulting from your transform method with the original dataset by plotting the points on an XY plot. Comment on the differences between original and transformed data in the cell directly below your plot. In your comment, explain why and how PCA can be used for dimensionality reduction

 

HINTS:

    You can use Scitkitlean as follows to calculate the new values of the data points in the original dataset when you reduce the dimensions of the data (from 2) to 1: 

pca = PCA(n_components=1)
pca.fit(data)
data_pca = pca.transform(data)
data_reduced = pca.inverse_transform(data_pca)

    You can use plots to compare the values in your original dataset with the dataset with reduced dimensionality:

plt.plot(data[:,0], data[:,1], 'or')
plt.plot(data_reduced[:,0], data_reduced[:,1],'xb')
plt.show()

    You can use your own PCA results to calculate the new values of the data points int the original dataset when you reduce the dimensions of the data (from 2) to 1:

reduced = np.dot(features[:,0],red_highvar.T)+mean.T

with:

    reduced: a 2x20 matrix of the new values of the dataset with dimensionality reduction applied
    features[:,0]: the 2x1 matrix (or column vector) which contains the Eigen vector associated with the highest variance
    red_highvar: a 20x1 matrix containing the reduced dataset which is the output of your transform method with n_components set to 1. 
    mean: a 1x2 matrix of the per-column mean values of your original data
    T: the transform operator as provided by Numpy
