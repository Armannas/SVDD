import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, LinAlgError
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.base import BaseEstimator, ClassifierMixin

class SVDD(BaseEstimator, ClassifierMixin):
    '''Support Vector Data Descriptor.'''
    def __init__(self, kernel_type='rbf', bandwidth=1, order=2, fracrej=np.array([0.1, 1])):
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.order = order
        self.fracrej = fracrej

    def _more_tags(self):
        return {'binary_only': True}

    def compute_kernel(self, X, Z):
        """
        Compute kernel for given data set.
        Parameters
        ----------
        X : array
            data set (N samples by D features)
        Z : array
            data set (M samples by D features)
        type : str
            type of kernel, options: 'linear', 'polynomial', 'rbf',
            'sigmoid' (def: 'linear')
        order : float
            degree for the polynomial kernel (def: 2.0)
        bandwidth : float
            kernel bandwidth (def: 1.0)
        Returns
        -------
        array
            kernel matrix (N by M)
        """

        # Data shapes
        # N, DX = X.shape

        # Only RBF kernel is implemented
        if self.kernel_type != 'rbf':
            raise NotImplementedError('Kernel not implemented yet.')
        else:
            # Radial basis function kernel
            return np.exp(-cdist(X, Z, 'sqeuclidean') / (self.bandwidth ** 2))

        # These kernels are not implemented yet
        # # Select type of kernel to compute
        # if self.kernel_type == 'linear':
        #     # Linear kernel is data outer product
        #     return np.dot(X, Z.T)
        # elif self.kernel_type == 'polynomial':
        #     # Polynomial kernel is an exponentiated data outer product
        #     return (np.dot(X, Z.T) + 1) ** self.order
        # elif self.kernel_type == 'sigmoid':
        #     # Sigmoidal kernel
        #     return 1. / (1 + np.exp(np.dot(X, Z.T)))


    def optimize_SVDD(self, X, y, c):
        # Solve (quadratic) SVDD program.
        # Maximize Lagrangian by minimizing negation:
        # a'.T @ K @ a' - diag(K).T @ a' (Eq. 2.28)

        # Subject to  0 <= a' <= C  (Box constraints, Eq. 2.20)
        #              1.T a'  = 1 (Equality constraint, Eq. 2.6 and 2.18)

        # where a' = y * a (Eq. 2.22)

        # CVXOPT has following format:
        # https://github.com/cvxopt/cvxopt/blob/cc46cbd0cea40cdb8c6e272d3e709c268cd38468/src/python/coneprog.py#L4156
        # Minimize   (1/2) x.T P x + q.T x
        #
        # Subject to G x <= h
        #            A x  = b

        # TODO: wait, I am not actually substituting a' in the box constraints
        # TODO: Is this valid!!???
        # Therefore, box constraints are rewritten to:
        #                 -I a' <=  0
        #                  I a' <=  C
        # Where I is the identity matrix

        # CVXOPT does not accept integers
        y = np.double(y)

        K = self.compute_kernel(X, X)
        # self.K = K
        N = K.shape[0]
        # Incorporate labels y in kernel matrix to rewrite in standard form (again, Eq. 2.22)
        P = np.outer(y, y) * K

        q = matrix(- y * np.diag(P))
        # self.q = q
        # Regularize quadratic part if not positive definite
        i = -30
        posdef_warning = False
        I_matrix = np.eye(N)
        while not self.is_pos_def(P + (10.0 ** i) * I_matrix):
            if not posdef_warning:
                print('Warning: matrix not positive definite. Started regularization')
                posdef_warning = True

            i = i + 1
        P = P + (10.0 ** i) * I_matrix
        # self.P = P
        print("Finished regularization")
        P = matrix(2*P)

        lb = np.zeros(N)

        # Either assign C for every sample
        if len(c) > 2:
            ub = c
        # Or one for the inliers and outliers separately.
        else:
            ub = np.zeros(N)
            ub[y == 1] = c[0]
            ub[y == -1] = c[1]

        # Equality constraint
        A = matrix(y, (1, N), 'd')
        b = matrix(1.0)

        # Box constraints written as inequality constraints
        # With correct substitution of alpha incorporating labels
        #  = matrix(np.vstack([-np.eye(N)*y, np.eye(N)*y]), (2 * N, N))

        G = matrix(np.vstack([-np.eye(N), np.eye(N)]), (2 * N, N))
        h = matrix(np.hstack([lb, ub]), (2 * N, 1))

        # Find optimal alphas
        solvers.options['show_progress'] = False
        res = solvers.qp(P, q, G, h, A, b)

        alfs = np.array(res['x']).ravel()

        # TODO: Figure out if the multiplication by one in the paper is valid
        # % Important: change sign for negative examples:
        alfs = y * alfs

        # The support vectors:
        SV_inds = np.where(abs(alfs) > 1e-8)[0]

        # Eq. 2.12, second term: Distance to center of the sphere (ignoring the offset):
        self.Dx = -2 * np.sum(np.outer(np.ones(N), alfs) * K, axis=1)

        # Support vectors where 0 < alpha_i < C_i for every i
        borderx = SV_inds[np.where((alfs[SV_inds] < ub[SV_inds]) & (alfs[SV_inds] > 1e-8))[0]]
        if np.shape(borderx)[0] < 1:
            borderx = SV_inds

        # Although all support vectors should give the same results, sometimes
        # they do not.
        self.R2 = np.mean(self.Dx[borderx])

        # Set all nonl-support-vector alpha to 0
        alfs[abs(alfs) < 1e-8] = 0.0

        return (alfs, self.R2, self.Dx, SV_inds)

    def fit(self, X, y):
        # Setup the appropriate C's
        self.C = np.zeros(2)
        nrtar = len(np.where(y == 1)[0])
        nrout = len(np.where(y == -1)[0])
        
        # we could get divide by zero, but that is ok.
        self.C[0] = 1/(nrtar*self.fracrej[0])
        self.C[1] = 1/(nrout*self.fracrej[1])

        # Only for comparison test with Matlab version
        # self.C = np.array([1, 1])
        alfs, R2, Dx, SV_inds = self.optimize_SVDD(X, y, self.C)

        self.SVx = X[SV_inds, :] # (data corresponding to support vectors)
        self.SV_alfs = alfs[SV_inds] # support vectors
        self.alfs = alfs
        self.SV_inds = SV_inds

        # Compute the offset (Eq. 2.12, First and last term):
        # Not essential, but now gives the possibility to
        # interpret the output as the distance to the center of the sphere.
        self.offs = 1 + self.SV_alfs.T @ self.compute_kernel(self.SVx, self.SVx) @ self.SV_alfs
        # Eq. 2.12: Include second term to determine threshold
        self.threshold = self.offs + R2

        # Only for comparison test with Matlab version
        # return (alfs, R2, Dx, SV_inds)

        # Scikit learn requires to return the classifier instance.
        return self

    def predict(self, X):
        # # Number of test samples
        # self.m = np.shape(X)[0]
        #
        # # Eq. 2.11: Distance of test objects to center of sphere
        # K = self.compute_kernel(X, self.SVx)
        # self.out = self.offs - 2 * K @ self.SV_alfs
        self.out = self.decision_function(X)

        self.newout = np.vstack([self.out, -1 * np.ones(self.m) * self.threshold])

        # Eq. 2.14: Indicator function
        I_out = np.greater_equal(self.newout[0, :], self.newout[1, :]).astype(int)

        # Assign predicted labels
        I_out[I_out == 0] = -1

        return I_out
    def decision_function(self, X):
        # Number of test samples
        self.m = np.shape(X)[0]

        # Eq. 2.11: Distance of test objects to center of sphere
        K = self.compute_kernel(X, self.SVx)
        out = self.offs - 2 * K @ self.SV_alfs

        return -1 * out

    def is_pos_def(self, X):
        """Check for positive definiteness."""
        try:
            cholesky(X)
            return True
        except LinAlgError:
            return False

    def _plot_contour(self, X, y, lims=None):

        fig, ax = plt.subplots()
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='+', zorder=2, label='target')
        ax.scatter(X[y == -1, 0], X[y == -1, 1], facecolors='none', edgecolors='r', zorder=2, label='outlier')

        # If train data is included in plot, mark the support vectors
        if self.SVx[0] in X:
            ax.scatter(X[self.SV_inds, 0], X[self.SV_inds, 1], facecolors='none', edgecolors='white', zorder=2, label='support vector')

        # Define plot boundaries manually
        if lims:
            ax.set_xlim(lims[0:2])
            ax.set_ylim(lims[2:])

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        delta = 0.025
        # x1 = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
        # x2 = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
        x1 = np.arange(xmin, xmax, delta)
        x2 = np.arange(ymin, ymax, delta)

        X1, X2 = np.meshgrid(x1, x2)
        X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

        K = self.compute_kernel(X_grid, self.SVx)
        Z = (self.offs - 2 * K @ self.SV_alfs).reshape(np.shape(X1))
        ax.contourf(X1, X2, Z, cmap=cm.gray, levels=50)

        # Draw decision line
        ax.contour(X1, X2, Z, levels=(self.threshold,), colors='white', linestyles='dashed', zorder=1)
        plt.legend(facecolor='grey')
        plt.title("SVDD contour map")
        plt.savefig('test.pdf', bbox_inches='tight', pad_inches=0)

        return plt
