from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import numpy as np

class KNNClassifier:
    def __init__(self, ks, n_components=None, n_splits=None,
                 weights="uniform", algorithm="brute", metric="minkowski", p=2):
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.ks = ks
        self.n_components = n_components
        self.n_splits = n_splits
        self.weights = weights
    

    def pca_on_train_and_test(self, n_components, X_train, X_test):
        pca = PCA(n_components=n_components)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)
        return X_train_transformed, X_test_transformed
    

    def train_and_predict(self, k, X_train, Y_train, X_test, Y_test, n_component=None):
        
        """
        Create a knn classifier with k nearest neighbor
        Returns:
            count_error: number of errors
            predicted_labels: predicted labels
        """
        knn_classifier = KNeighborsClassifier(n_neighbors=k,
                                                algorithm=self.algorithm,
                                                metric=self.metric,
                                                p=self.p,
                                                weights=self.weights)
        
        if n_component:
            X_train, X_test = self.pca_on_train_and_test(n_component, X_train, X_test)

        knn_classifier.fit(X_train, Y_train)
        predicted_labels = knn_classifier.predict(X_test)
        count_error = np.count_nonzero(predicted_labels != Y_test)
        return count_error, predicted_labels


    def kf_choose_k(self, X, Y, n_component=None):
        
        kf = KFold(n_splits=self.n_splits)
        error_rates = []

        for k in self.ks:
            # Error list for each k
            errors_k = []

            for train_idx, val_idx in kf.split(X):
                # Split
                X_train_kf, Y_train_kf = X[train_idx], Y[train_idx]
                X_val_kf, Y_val_kf = X[val_idx], Y[val_idx]

                # Do KNN
                count_error, _ = self.train_and_predict(k, X_train_kf, Y_train_kf,
                                                        X_val_kf, Y_val_kf, n_component=n_component)
                errors_k.append(count_error)
            
            error_rates.append(np.average(errors_k)/Y_val_kf.shape[0])
        return error_rates
    

    def pca_and_kf_choose_k(self, X, Y):
        """
        Return:
            errors: [n_components x ks]
        """
        errors_pca = []
        for n_component in self.n_components:
            error_rates = self.kf_choose_k(X, Y, n_component)
            errors_pca.append(error_rates)
        return errors_pca
    

    def average_error_on_training(self, X, Y, n_component=None):
        error_rates = []
        for k in self.ks:
            # k + 1 because the query point itself will be counted as a neighbor
            error, _ = self.train_and_predict(k + 1, X, Y, X, Y, n_component) 
            error_rates.append(error / Y.shape[0])
        
        return error_rates
    


