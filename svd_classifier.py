from numpy.linalg import svd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels

class SVDClassifier(BaseEstimator, ClassifierMixin):
    """
    Classification using SVD
    
    Singular value decomposition for A:
    A = (U)(S)(V^T)
    Columns of U form orthogonal basis of column space of A
    Given a target vector v, 'closeness' to it being a linear combination of
    column vectors of A can be approximated by norm:
    ||v-((Uk)((Uk)^T))v|| where Uk represents first k columns of U (i.e. a 
    low rank approximation)
    
    Given a set of trainging examples falling in categories {c_1, c_2,...,c_k}
    For each c_i construct, A_{c_i} by stacking columns of training examples 
    corresponding to category c_i.
    Given a vector v in the test set, find the closeness to each A_{c_i}
    as described above. This gives a approximate measure of how well vector v 
    can be represented as a linear combination of training examples of a 
    category. Return the c_j that gives the closest representation of v.
    
    [Ref] : SVD based algo - https://mazack.org/papers/mazack_masters.pdf
    """
    def __init__(self, k=15):
        # k represents the number of columns to select from left singular vector
        self.k = k
        # stores the projection matrix to compute residuals to each categoriy's
        # left basis vectors
        # (I-((Uk)((Uk)^T)))
        self.cat_residuals = None

    def fit(self, X, y=None):
        """
        Fit model according to training data
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.   
            
        Returns
        -------
        self : object
            An instance of the estimator.
        """            

        X, y = check_X_y(X, y)
        self.categories = unique_labels(y)
        
        def get_approx_basis(X_train, y_train, category):
            # get examples with category 'category'
            cat_indx = [y == category for y in y_train] 
            cat_X = X_train[cat_indx]
            # Form matrix A by stacking examples of category as its columns
            A = np.column_stack(cat_X)
            # SVD
            U, S, Vtranspose = svd(A)
            # left_basis = k left singular vectors
            left_basis = U[:,:self.k]
            return left_basis
        # find basis for each categories
        category_basis_matrices = [get_approx_basis(X, y, k) for k in 
                                   self.categories]
        # store projection matrix to compute residuals for every category
        self.cat_residuals = [np.identity(X[0].shape[0]) - 
                              np.matmul(category_basis_matrices[k], 
                                        category_basis_matrices[k].T) 
                              for k in self.categories]
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples_test, n_samples_train)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        assert self.cat_residuals is not None
        assert self.categories is not None
        X = check_array(X)
        def find_closest(test_value, pr_residuals):
            # find closest U_k to test_value using norm and return the 
            # corresponding label
            closest = np.inf
            label = None
            for cat, pr in zip(self.categories, pr_residuals):
                t_norm = np.linalg.norm(np.matmul(pr, test_value))
                if t_norm < closest:
                    closest = t_norm
                    label = cat
            return label
        y_pred = [find_closest(tv, self.cat_residuals) for tv in X]
        return y_pred
