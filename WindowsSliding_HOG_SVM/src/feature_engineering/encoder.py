import numpy as np

# One hot encoder
class OneHotEncoder():
    
    def __innit__(self):
        
        self.__class_list = None
        self.__index_list__ = None
        self.vocab = None
        self.vocab_size = 0
        
     
    def fit(self, Y:list):
        
        self.__class_list = list(set(Y))
        self.__index_list = range(len(self.__class_list))
        
        self.vocab = dict(zip(self.__index_list, self.__class_list))
        self.vocab_size = len(self.__class_list)
        
    
    def transform(self, Y:list):
        
        if self.__class_list:
            Y_encoded = []
    
            for y in Y:
            
                index = self.__class_list.index(y)
                y_encoded = np.zeros((len(self.__class_list), 1))
                y_encoded[index, 0] = 1
                Y_encoded.append(y_encoded)
            
            return Y_encoded
        
        else:
            raise Exception("Class object is not fitted.")
        
        
    def fit_transform(self, Y:list):
        self.fit(Y)
        return self.transform(Y)
        
    
    def reverse_transform(self, Y:list):
        
        if self.__class_list:
            Y_decoded = []
            
            for y in Y:
                key = np.argmax(y)
                Y_decoded.append(self.vocab[key])
            
            return Y_decoded
            
        else:
            raise Exception("Class object is not fitted.")