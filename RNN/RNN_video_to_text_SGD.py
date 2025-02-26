import numpy as np


# Preprocess
def Extractor(data:np.array, filter:np.array)->np.array:
    # Notes here
    conv_data = np.zeros(data.shape)
    half_size = filter.shape[0] // 2
    sheet = np.zeros((data.shape[0] + half_size*2, data.shape[1] + half_size*2))
    sheet[half_size:sheet.shape[0]-half_size, half_size:sheet.shape[1]-half_size] = data
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window = sheet[i:i+filter.shape[0], j:j+filter.shape[1]]
            conv_data[i,j] = (window * filter).sum()
            
    return conv_data


def flatten(data:np.array)->np.array:
    return data.reshape(data.size,1)


class WordEncoder():
    
    def __init__(self, max_n_window:int, n_words:int=1):
        
        self.vocab = {0:"ptyem"}
        self.vocab_size = 1 # Size include 'empty'
        self.__n = n_words
        self.__txtlen = max_n_window*n_words
        self.__fitted = False
        
        
    def __words_list__(self, Y:list):
        
        words_list = []
        for text in Y:
            words = text.split()
            size_remain = self.__txtlen - len(words)
            text_full = words + ["ptyem"]*size_remain
        
        return words_list
        
        
    def __check_exist__(words:list):
        return words in vocab.values()


    def __one_hot_encoder__(self, vectorize_window:int):
        
        vector = np.zeros(len(self.vocab_size),1)
        vector[vectorize_window] = 1
        
        return vector

        
    def fit(self, Y:list):
        
        wl = self.__words_list__(Y)
        for word_index in range(len(wl)-self.__n+1):
            current_vocab_index = len(self.vocab.keys())
            words = []
            i = 0
            while i < self.__n:
                words.append(wl[word_index + i]
                i++
            if not self.__check_exist__(words):
                vocab[current_vocab_index+1] = words
                self.vocab_size++
        self.__fitted = True
    
    
    def transform(self, Y:list)->list:
        
        if self.__fitted:
            
             Y_vectorized = []
             inverse_vocab = {value:key for key, value in self.__vocab.item()}
            
            for text in Y:
                vectorize_str = np.zeros(self.vocab_size, self.__txtlen/self.__n)
                words = text.split()
                words += [""]*(len(words)%self.__n)
                while i < len(words):
                    window = [words[i+ w] for w in range(self.__n)]
                    vectorize_window = inverse_vocab[window]
                     vector = __one_hot_encoder__(vectorize_window)
                     vectorize_str[:, i/self.__n] = vector
                    i += self.__n
                Y_vectorized.append(vectorize_str)
            return Y_vectorized
                    
        else:
            raise Exception("Encoder object is not fitted")
        
        
    def fit_transform(self, Y:list):
        
        self.fit(Y)
        return self.transform(Y)
    
    
    def reverse_transform(self, Y_encoded:list)->list:
        
        if self.__fitted:
            Y_decoded = []
            for matrix in Y_encoded:
                words = []
                for vector in range(self.__txtlen/self.__n):
                    index = np.where(vector==1)[0][0]
                     words.append(*vocab[index])
                Y_decoded.append(' '.joint(words))
            return Y_decoded
        
        else:
            raise Exception("Encoder object is not fitted")
            

# Activation functions
def ReLU(z:np.array)->np.array:
    return np.maximum(z, 0)


def SoftMax(z:np.array)->np.array:
    exponential = np.exp(z)
    sum = exponential.sum()
    return exponential/sum


# Model
class VideoToScript():
    
    def __init__(self, input_size:int, output_size:int, n_hid_n:int, epochs:int, script_size:int=20, lr:float=0.001):
        
        # Initialize parameters
        # Encoder
        self.__Wx = np.random.rand(n_hid_n, input_size)
        self.__Wa1 = np.random.rand(n_hid_n, n_hid_n)
        self.__bh1 = np.random.rand(n_hid_n, 1)
        
        # Decoder
        self.__Wa2 = np.random.rand(n_hid_n, n_hid_n)
        self.__bh2 = np.random.rand(n_hid_n, 1)
        
        # Output
        self.__Wy = np.random.rand(output_size, n_hid_n)
        self.__by = np.random.rand(output_size, 1)
        
        # Architecture
        self.__input_size = input_size
        self.__nn = n_hid_n
        self.__output_size = output_size
        self.__lr = lr
        self.__epochs = epochs
        self.__sz = script_size
        
        # data
        self.__X = []
        self.__Y = []
        
        # Archive
        self.__A1 = []
        self.__A2 = []
        self.__yhat = []
        
     
    def __encoder__(self, index):
        
        a1 = np.zeros(self.__nn, 1) # initial zero memory
        time = self.__X[index].shape[1]
        self.__A1.append(a1)
        
        for t in range(time):
            z1 = np.dot(self.__Wx, self.__X[index][:, t]) + np.dot(self.__Wa1, a1) + self.__bh1
            a1 = ReLU(z1)
            self.__A1.append(a1)
        
        self.__A2.append(a1)
       
        
    def __decoder__(self):
         
         ypred = np.zeros(self.__output_size, self.__script_size)
         
         for word in range(len(self.__script_size)):
             z2 = np.dot(self.__Wa2, self.__A1[-1]) + self.__bh2 # Input the final output of the encoder
             a2 = ReLU(z2)
             self.__A2.append(a2)
             z3 = np.dot(self.__Wy, a2) + self.__by
             ypred[:, word] = SoftMax(z3)
             
         self.__yhat.append(ypred)
      
      
    def __forward_propagation__(self, index):
          
          self.__encoder__(index) # Encode time frames
          self.__decoder__()      # Decode into script text labels
      
      
    def __backward_propagation(self, index):
          pass
          
          
    def fit(self, X:list, Y:list):
          
          # Assign data
          self.__X = X
          self.__Y = Y
          
          # Train model
          for index in range(len(self.__X)):
              self.__forward_propagation__(index)
          
          print(self.__yhat)
      

def main():
    X1 = np.random.randint(0, 255, (17, 20, 20))
    y1 = "There is a house."
    
    fltr = np.asarray([
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
    ])
    
    X1_conv = Extractor(X1[1,:,:], fltr)
    print(X1_conv)
    
main()