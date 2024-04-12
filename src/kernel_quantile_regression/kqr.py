
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
from sklearn.metrics.pairwise import rbf_kernel

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

    gamma : int, default='1'
        Bandwith of rbf kernel

    C : int, default='0.5'
        the cost regularization parameter. This parameter controls the smoothness of the fitted function, essentially higher values for C lead to less smooth functions

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The dependent variable, our target :meth:`fit`.
    """
    def __init__(self, alpha, C=1, gamma=1):
        self.gamma = gamma
        self.C=C
        self.alpha=alpha

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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.X_ = X

        self.y_ = y
        # build convex optimisation problem
        K=rbf_kernel(self.X_, gamma=self.gamma)
        # the 0.5 in front in the optimisation probelm is taken into account by cvxopt library
        K = matrix(K)
        # multiply by one to convert matrix items to float https://stackoverflow.com/questions/36510859/cvxopt-qp-solver-typeerror-a-must-be-a-d-matrix-with-1000-columns
        r=matrix(y)* 1.0
        # equality constraint
        A = matrix(np.ones(y.size)).T
        b = matrix(1.0)
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
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # compute y with a and b
        # self.X_=X_train
        # X=X_pred/X_test
        K=rbf_kernel(self.X_, X, gamma=self.gamma)
        y_pred=self.a.T@K +self.b
        return y_pred


def cvx_solver(K,y,G,h,A,b):
    sol = qp(P=K,q=-y,G=G,h=h,A=A,b=b)
    return sol

if __name__=="__main__":
    # load data
    df=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/temperatures_melbourne.csv", sep=";", decimal=".")

    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df["Yt-1"], df["Yt"], test_size=0.2, random_state=4)

    # equally space dataset for plotting
    eval_set=np.linspace(df["Yt-1"].min(), df["Yt-1"].max(), 100).T

    # reshape data
    X_train= X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    eval_set=eval_set.reshape(-1,1)

    # scale data because kernel methods need it to work better
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    eval_set_scaled = scaler.transform(eval_set.reshape(-1, 1))
    
    # add column of ones
    # ones_column = np.ones((X_train_robust.shape[0], 1))
    # # X_train_robust = np.hstack((ones_column, X_train_robust))
    # ones_column_eval = np.ones((eval_set_robust.shape[0],1))
    # eval_set_robust = np.hstack((ones_column_eval, eval_set_robust))
    # q=0.1

    avg_pinball=0
    # plot
    plt.plot(X_train,y_train,"o", alpha=0.2)

    for q in tqdm(quantiles):
    #     neg_mean_pinball_loss_scorer = make_scorer(
    #     mean_pinball_loss,
    #     alpha=q,
    #     greater_is_better=False,  # maximize the negative of the loss
    #     )
    #     param_grid_kqr = dict(
    #     C=[1e10],
    #     gamma=[0.1,0.5,1,5]
        
    # )   
        # kqr_blueprint=KQR(alpha=q)
        # kqr=HalvingRandomSearchCV(
        #         kqr_blueprint,
        #         param_grid_kqr,
                
        #         scoring=neg_mean_pinball_loss_scorer,
        #         n_jobs=None
            # ).fit(X_train_robust, y_train.ravel())
        # print("best tuned hyperparameters: ", kqr.best_params_)
        kqr=KQR(alpha=q, gamma=1, C=1e3).fit(X_train_scaled, y_train.ravel())

        # best model prediction
        y_train_predr=kqr.predict(X_train_scaled)
        y_test_predr=kqr.predict(X_test_scaled)
        # pinball loss
        print("pinball loss" ,mean_pinball_loss(y_test,y_test_predr, alpha=q))
        avg_pinball+=mean_pinball_loss(y_test,y_test_predr, alpha=q)

    # plot
        L = sorted(zip(eval_set_scaled,kqr.predict(eval_set_scaled)), key=itemgetter(0))
        eval_set_scaled, y_eval_predr = zip(*L)

        plt.plot(eval_set,y_eval_predr, alpha=0.4,label=f"q={q}",color="black", linestyle="dashed")
    
    plt.legend()
    plt.title("Kernel quantile regression")
    plt.show()

    print("summed pinball", avg_pinball)
    
    # n=3000
    # Q = np.random.randn(n,n)
    # Q = 0.5 * (Q + Q.T)
    # Q = Q + n * np.eye(n)
    # Q = matrix(Q)
    # p = matrix(np.random.sample(n))
    # G1 = matrix(np.eye(n))
    # h1= matrix(10000*np.ones(n))
    # G2 = matrix(- np.eye(n))
    # h2= matrix(-10000*np.ones(n))
    # # concatenate
    # G = None
    # h = None
    # A = matrix(np.ones(n)).T
    # b = matrix(1.0)
    # cvx_solver(Q,p,G,h,A,b)


    # df2=pd.read_csv("/Users/luca/Desktop/ThesisKernelMethods/dataset_applications/simple_df.csv", sep=",", decimal=".")
    # print(df2)
    # x =df2['x']
    # x=x.reshape(-1, 1)
    # y =df2['y']
    
    # kqr = KQR().fit(x, y)
    # kqr = KQR().fit(X_train, y_train)
    # # y_test_pred_qr=kqr.predict(eval_set)
