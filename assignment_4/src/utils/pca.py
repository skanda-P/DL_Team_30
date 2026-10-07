# PCA utility to compute eigenvectors and mean vectors on training data and project train, validation, and test sets.
import numpy as np


class PCA:
    def __init__(self):
        self.mean_ = None
        self.eigenvalues_ = None     # (D,)   descending
        self.components_ = None      # (D, D) column j is the j-th eigenvector

    def fit(self, X_train):
        """
        Eigendecomposition of the training covariance matrix. All D components are
        computed once; transform(X, k) then keeps the top k, so 32/64/128/256 share
        a single decomposition.
        """
        X = np.asarray(X_train, dtype=np.float64)
        n = X.shape[0]

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (n - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)      # ascending, orthonormal columns
        order = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = np.clip(eigvals[order], 0.0, None)
        self.components_ = eigvecs[:, order]
        return self

    def transform(self, X, k):
        """Mean-subtract with the TRAINING mean, project on the top-k eigenvectors."""
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_[:, :k]

    def inverse_transform(self, Z):
        """Reconstruct (D-dim) inputs from the k-dim projections."""
        k = Z.shape[1]
        return Z @ self.components_[:, :k].T + self.mean_

    def explained_variance_ratio(self, k):
        """Fraction of the training variance retained by the top-k components."""
        return float(self.eigenvalues_[:k].sum() / self.eigenvalues_.sum())