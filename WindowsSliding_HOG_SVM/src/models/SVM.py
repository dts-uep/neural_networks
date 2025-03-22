import numpy as np


class SVM():
    
    def __init__(self, n_class:int, input_size:int):
        
        self.__W = np.random.rand(n_class, input_size+1) / (input_size + 1)*10 # Get initial of around 0 to 1
        self.__X = None
        self.__Y = None
        self.__Z = None
        self.__loss = 0
        self.__c = n_class
        self.__lr = None
        self.input_size = input_size
    
    
    def __forward__(self):
        self.__Z = np.dot(self.__W, self.__X) 
    
    
    def __update_weight__(self):
        
        # Calculate loss
        Ztrue = np.where(self.__Y==1, self.__Z, 0).sum(axis=0)
        Loss_matrix = 1 - Ztrue + self.__Z 
        Loss_matrix = np.maximum(Loss_matrix, 0)
        Loss_matrix = np.where(self.__Y == 1, 0, Loss_matrix)
        self.__loss = Loss_matrix.sum()
        
        del Ztrue
        
        # Get gradients
        gradient_count = np.where(Loss_matrix > 0, 1, 0)
        gradient_count = np.where(self.__Y == 1, -1, gradient_count)
        
        gradient_matrix = np.zeros_like(self.__W)
        for w in range(self.__c):
            gradient_matrix[w, :] = (self.__X*gradient_count[w,:]).sum(axis=1).T
        
        del gradient_count
        
        # Update parameters
        self.__W -= self.__lr*gradient_matrix / (self.__X.shape[1]+1) # Scale down by batch size and origin down_scale ratio
        
    
    def fit(self, X:list, Y:list, epochs:int=1, lr:float=0.001, stop:float=0.0000001):
        
        # Set up
        self.__X = np.hstack(X)
        self.__X = np.vstack([self.__X, np.ones((1, len(X)))])
        self.__lr = lr
        
        self.__Y = np.hstack(Y)
        previous_loss = 100000

        for _ in range(epochs):
            
            self.__forward__()
            self.__update_weight__()
            
            print(f"Loss: {self.__loss}")
            if previous_loss - self.__loss <= stop:
                print("Stop")
                break
            previous_loss = self.__loss
     

    def predict(self, X:list):
        
        if self.__lr:

            self.__X = np.hstack(X)
            self.__X = np.vstack([self.__X, np.ones((1, len(X)))])
            self.__forward__()
            
            return self.__Z
        
        else:
            
            raise Exception("Class model is not fitted.")
        
    
    def predict_label(self, X:list):
        
        y_label = []
        y_pred = self.predict(X)
        
        for n in range(len(X)):
            highest_score = np.argmax(y_pred[:, n])
            label = np.zeros((self.__c, 1))
            label[highest_score] = 1
            y_label.append(label)
        
        return y_label
