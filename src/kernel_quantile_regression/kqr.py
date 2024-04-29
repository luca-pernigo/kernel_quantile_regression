
import cvxopt as opt
from cvxopt.solvers import qp, options
from cvxopt import matrix, spmatrix, sparse
from cvxopt import blas
 
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import pandas as pd

from operator import itemgetter

from scipy.stats import uniform
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.experimental import enable_halving_search_cv

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss

from sklearn.gaussian_process.kernels  import Matern
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import sigmoid_kernel

from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


import sys
import time
from tqdm import tqdm


class KQR(RegressorMixin, BaseEstimator):
    """ Code implementing kernel quantile regression.

    Parameters
    ----------
    alpha : int, default='0.5'
        quantile under consideration

    kernel_type : str, default='gaussian_rbf'
        kind of kernel function    
    
    gamma : float, default=1
        bandwith parameter of rbf gaussian, laplacian, sigmoid, chi_squared, matern, periodic kernels
    
    sigma : float, default=None
        additional parameter for kernels taking more than one parameter
    
    omega : float, default=None
        additional parameter for kernels taking more than two parameters

    c : float, default=None
        constant offset added to scaled inner product of polynomial, sigmoid kernels

    d : float, default=None
        degree of polynomial kernel

    nu : float, default=None
        nu parameter of matern kernel
    
    p : float, default=None
        period of periodic kernel

    C : int, default='0.5'
        the cost regularization parameter. This parameter controls the smoothness of the fitted function, essentially higher values for C lead to less smooth functions

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The dependent variable, our target :meth:`fit`.
    """
    def __init__(self, alpha, kernel_type="gaussian_rbf", C=1, gamma=1, sigma=None, omega=None, c=None, d=None, nu=None, p=None):
    
        self.C=C
        self.alpha=alpha
        self.kernel_type=kernel_type

        # prm_s
        self.gamma=gamma
        self.sigma=sigma
        self.omega=omega
        self.c=c
        self.d=d
        self.nu=nu
        self.p=p

    def kernel(self, X, Y):
        # kernels according to specfied kernel type
        if self.kernel_type=="gaussian_rbf":
            return rbf_kernel(X,Y, gamma=self.gamma)
        
        elif self.kernel_type=="laplacian":
            return laplacian_kernel(X,Y, gamma=self.gamma)
        
        
        elif self.kernel_type=="linear":
            return linear_kernel(X,Y)

        elif self.kernel_type=="cosine":
            return cosine_similarity(X,Y)
        
        elif self.kernel_type=="polynomial":
            return polynomial_kernel(X,Y, coef0=self.c, degree=self.d)
        
        elif self.kernel_type=="sigmoid":
            return sigmoid_kernel(X,Y, coef0=self.c, gamma=self.gamma)
        
        elif self.kernel_type=="matern":
            matern_kernel=1.0*Matern(length_scale=self.gamma, nu=self.nu)
            return matern_kernel(X,Y)

        elif self.kernel_type=="chi_squared":
            return chi2_kernel(X,Y,gamma=self.gamma)
        
        elif self.kernel_type=="periodic":
            periodic=1.0*ExpSineSquared(length_scale=self.gamma, periodicity=self.p)
            return periodic(X,Y)
        
        # class of kernels functions are closed under addition and product
        elif self.kernel_type=="gaussian_rbf_x_laplacian":
            return rbf_kernel(X,Y, gamma=self.gamma)* laplacian_kernel(X,Y, gamma=self.sigma)
        
        
         # else not implemented
        else:
            raise NotImplementedError('No implementation for selected kernel')
        
        
        

    def fit(self, X, y):
        """Implementation of fitting function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # check that X and y have correct shape
        # self.kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)

        X, y = check_X_y(X, y)
        
        self.X_ = X

        self.y_ = y
        # build convex optimisation problem
        K=self.kernel(self.X_,self.X_)
        # the 0.5 in front in the optimisation probelm is taken into account by cvxopt library
        K = matrix(K)
        # multiply by one to convert matrix items to float https://stackoverflow.com/questions/36510859/cvxopt-qp-solver-typeerror-a-must-be-a-d-matrix-with-1000-columns
        r=matrix(y)* 1.0
        # equality constraint
        A = matrix(np.ones(y.size)).T
        b = matrix(0.0)
        # two inequality constraints
        G1 = matrix(np.eye(y.size))
        h1= matrix(self.C*self.alpha*np.ones(y.size))
        G2 = matrix(- np.eye(y.size))
        h2= matrix(self.C*(self.alpha-1)*np.ones(y.size))
        # concatenate
        G = matrix([G1,G2])
        h = matrix([h1,-h2])
        # Solve
        sol = qp(P=K,q=-r,G=G,h=h,A=A,b=b)
        # alpha solution
        self.a=np.array(sol["x"]).flatten()
        
        # see coefficients
        # print("coefficients a: ",self.a)

        # check that summation equality to one holds
        # print("coefficients sum up to 1:", np.sum(self.a))
        
        # condition, index set of support vector
        squared_diff = (self.a - (self.C * self.alpha))**2 + (self.a - (self.C * (self.alpha - 1)))**2

        # get the smallest squared difference
        offshift = int(np.argmin(squared_diff))
        # print(offshift)
        
        # calculate bias term b
        self.b = y[offshift] - self.a.T@K[:,offshift]
        # print("beta mean: ", self.b)
        
        # Return the regressor
        return self

    def predict(self, X):
        """ Implementation of prediction function

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The model prediction for test data.
        """
        # check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # input validation
        X = check_array(X)

        # compute y with a and b
        # self.X_=X_train
        # X=X_pred/X_test
        K=self.kernel(self.X_, X)
        y_pred=self.a.T@K +self.b
        return y_pred


