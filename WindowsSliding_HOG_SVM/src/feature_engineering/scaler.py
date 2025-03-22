import numpy as np

class StandardScaler():


    def __init__(self):
        
        __variance_vector = None
        __mean_vector = None


    def fit(self, X:list):
        
        X = np.concatenate(X, axis=1)
        self.__variance_vector = np.var(X, axis=1).reshape((X.shape[0], 1))
        self.__mean_vector = np.mean(X, axis=1).reshape((X.shape[0], 1))


    def transform(self, X:list):
        
        X_scaled = []

        for vector in X:
            X_scaled.append((vector - self.__mean_vector) / self.__variance_vector)
        
        return X_scaled

    def fit_transform(self, X:list):
        
        self.fit(X)
        return self.transform(X)

    def reverse_transform(self, scaled_X:list):
        
        X_transformed = []

        for vector in scaled_X:
            X_transformed.append(vector * self.__variance_vector + self.__mean_vector)
        
        return X_transformed