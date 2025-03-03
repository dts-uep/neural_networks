import numpy as np
from tqdm import tqdm
import time

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
        
        self.vocab = {0:["ptyem"]*n_words}
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
            words_list += text_full
        
        return words_list
        
        
    def __check_exist__(self, words:list):
        return words in self.vocab.values()


    def __one_hot_encoder__(self, vectorize_window:int):
        
        vector = np.zeros((self.vocab_size,))
        vector[vectorize_window] = 1
        
        return vector

        
    def fit(self, Y:list):
        
        wl = self.__words_list__(Y)
        for word_index in range(len(wl)-self.__n+1):
            current_vocab_index = len(self.vocab.keys())-1
            words = []
            i = 0
            while i < self.__n:
                words.append(wl[word_index + i])
                i += 1
            if not self.__check_exist__(words):
                self.vocab[current_vocab_index+1] = words
                self.vocab_size += 1
        self.__fitted = True
    
    
    def transform(self, Y:list)->list:
        
        if self.__fitted:
            
            Y_vectorized = []
            idxes = list(self.vocab.keys())
            values = list(self.vocab.values())
            
            for text in Y:
                vectorize_str = np.zeros((self.vocab_size, int(self.__txtlen/self.__n)))
                words = text.split()
                size_remain = self.__txtlen - len(words)
                words = words + ["ptyem"]*size_remain
                i = 0
                while i < len(words):
                    window = [words[i+ w] for w in range(self.__n)]
                    vectorize_window = idxes[values.index(window)]
                    vector = self.__one_hot_encoder__(vectorize_window)
                    vectorize_str[:, int(i/self.__n)] = vector
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
                for vector in range(int(self.__txtlen/self.__n)):
                    index = np.where(matrix[:,vector]==1)[0][0]
                    words += self.vocab[index]
                    words = [w for w in words if w != "ptyem"]
                Y_decoded.append(' '.join(words))
            return Y_decoded
        
        else:
            raise Exception("Encoder object is not fitted")
            

# Activation functions
def ReLU(z:np.array)->np.array:
    return np.maximum(z, 0)


# ReLU cause gradient exploding, use tanh to avoid and with lower saturate rate than sigmoid
def tanh(z:np.array)->np.array:
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def SoftMax(z:np.array)->np.array:
    
    exponential = np.exp(z)
    sum = exponential.sum()
    
    return exponential/sum


# Loss function
def log_loss_each_point(y_true:np.array, y_pred:np.array):
    return  (-1 * y_true * np.log(y_pred)).sum()


def log_loss_average(Y_true:list, Y_pred:list):
    
    Loss = 0
    for data in range(len(Y_true)):
        Loss += log_loss_each_point(Y_true[data], Y_pred[data])
    Loss /= len(Y_true)
    
    return Loss
    

# Model
class VideoToScript():
    
    def __init__(self, input_size:int, output_size:int, n_hid_n:int, epochs:int, script_size:int=20, lr:float=0.001):
        
        # Initialize parameters
        # Encoder
        weight_scaler_e = 1
        weight_scaler_e = 1 / (input_size + n_hid_n)  # Comment this to unscale weights
        self.__Wx = np.random.rand(n_hid_n, input_size) * weight_scaler_e
        self.__Wa1 = np.random.rand(n_hid_n, n_hid_n) * weight_scaler_e
        self.__bh1 = np.random.rand(n_hid_n, 1) * weight_scaler_e
        
        # Decoder
        weight_scaler_d = 1
        weight_scaler_d = 1 / n_hid_n # Comment this to unscale weights
        self.__Wa2 = np.random.rand(n_hid_n, n_hid_n) * weight_scaler_d
        self.__bh2 = np.random.rand(n_hid_n, 1) * weight_scaler_d
        
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
        
        a1 = np.zeros((self.__nn, 1)) # initial zero memory
        time = self.__X[index].shape[1]
        self.__A1.append(a1)
        
        for t in range(time):
            z1 = np.dot(self.__Wx, self.__X[index][:, t, np.newaxis]) + np.dot(self.__Wa1, a1) + self.__bh1
            #a1 = ReLU(z1)
            a1 = tanh(z1)
            self.__A1.append(a1)
        
        self.__A2.append(a1) # Link
       
        
    def __decoder__(self):
         
        ypred = np.zeros((self.__output_size, self.__sz))
        a2 = self.__A1[-1] # Link as initial input
        
        for l in range(self.__sz):
            z2 = np.dot(self.__Wa2, a2) + self.__bh2 
            #a2 = ReLU(z2)
            a2 = tanh(z2)
            self.__A2.append(a2)
            z3 = np.dot(self.__Wy, a2) + self.__by
            ypred[:, l] = SoftMax(z3).reshape((self.__output_size,))
             
        self.__yhat.append(ypred)
      
      
    def __forward_propagation__(self, index):
          
        self.__encoder__(index) # Encode time frames
        self.__decoder__()      # Decode into script text labels
      
      
    def __backward_propagation__(self, index):
        
        # Gradient wrt Wx, Wa1, bh1 
        # Gradient of Loss wrt link
        gradlink_L = 0
        multiply_factor = np.ones_like(self.__Wa2)
        
        for l in range(self.__sz):  
            dLl_da2 = np.dot(self.__Wy.T ,(self.__yhat[index][:, l] - self.__Y[index][:, l]))[:, np.newaxis]
            multiply_factor = np.dot((self.__Wa2 * (1 - (self.__A2[l+1])**2)).T , multiply_factor)  # l+1 as not count link
            gradlink_L += np.dot(multiply_factor, dLl_da2)
        gradlink_L /= (self.__sz*self.__nn)
        
        # Gradinet of link wrt Wx, Wa1, bh1
        gradzT_L = gradlink_L * (1-tanh(self.__A1[-1])**2) # Mutual factor
        gradWx_L = 0
        gradWa1_L = 0
        gradbh1_L = 0
        multiply_factor = np.ones_like(self.__Wa1)
        
        for t in range(self.__X[index].shape[1]-1):
            gradWa1_L += np.dot(np.dot(multiply_factor.T, gradzT_L), self.__A1[-2-t].T)
            gradWx_L += np.dot(np.dot(multiply_factor.T, gradzT_L), self.__X[index][:, -1-t][:, np.newaxis].T)
            gradbh1_L += np.dot(multiply_factor.T, gradzT_L)
            
            multiply_factor = np.dot(multiply_factor, self.__Wa1*(1-tanh(self.__A1[-2-t])**2))
        gradWx_L /= ((self.__X[index].shape[1]-1)*self.__nn)
        gradWa1_L /= ((self.__X[index].shape[1]-1)*self.__nn)
        gradbh1_L /= ((self.__X[index].shape[1]-1)*self.__nn)
        
        # Update Wx, Wa1, bh1
        self.__Wx  -= self.__lr * gradWx_L
        self.__Wa1 -= self.__lr * gradWa1_L
        self.__bh1 -= self.__lr * gradbh1_L
        
        del gradWx_L, gradWa1_L, gradbh1_L, multiply_factor, gradzT_L, gradlink_L, dLl_da2
        
        # Gradient wrt Wa2, bh2
        gradWa2_L = np.zeros_like(self.__Wa2)
        gradbh2_L = np.zeros_like(self.__bh2)
        
        for l in range(self.__sz):
            gradz3_Li = (self.__yhat[index][:,l] - self.__Y[index][:, l])[:, np.newaxis]
            grada2_Li = np.dot(self.__Wy.T, gradz3_Li)
            gradz2_Li = grada2_Li * (1-tanh(self.__A2[l+1]))
            
            gradWa2_Li = np.dot(gradz2_Li, self.__A2[l].T)
            gradWa2_L += gradWa2_Li
            
            gradbh2_Li = gradz2_Li
            gradbh2_L += gradbh2_Li
        gradWa2_L /= self.__sz
        gradbh2_L /= self.__sz
        
        # Update Wa2, bh2
        self.__Wa2 -= self.__lr * gradWa2_L
        self.__bh2 -= self.__lr * gradbh2_L
        
        del gradWa2_L, gradbh2_L, gradz3_Li, grada2_Li, gradz2_Li, gradWa2_Li, gradbh2_Li
        
        # Gradient wrt Wy, by
        gradWy_L = np.zeros_like(self.__Wy)
        gradby_L = np.zeros_like(self.__by)
        
        for l in range(self.__sz):
            gradz3_Li = (self.__yhat[index][:,l] - self.__Y[index][:, l])[:, np.newaxis]
            
            gradWy_li = np.dot(gradz3_Li, self.__A2[l+1].T)
            gradWy_L += gradWy_li
            
            gradby_Li = gradz3_Li
            gradby_L += gradby_Li
        gradWy_L /= self.__sz
        gradby_L /= self.__sz
        
        # Update Wy, by
        self.__Wy -= self.__lr * gradWy_L
        self.__by -= self.__lr * gradby_L
          
          
    def fit(self, X:list, Y:list):
          
        # Assign data
        self.__X = X
        self.__Y = Y
          
        # Train model
        for epoch in range(self.__epochs):
            sum_loss = 0
            with tqdm(total=len(X), ncols=80) as pbar:
                for index in range(len(self.__X)):
                    self.__forward_propagation__(index)
                    loss = log_loss_each_point(Y[index], self.__yhat[index])
                    sum_loss += loss
                    self.__backward_propagation__(index)
                    self.__A1 = []
                    self.__A2 = []
                    # Format progress bar
                    pbar.set_description(f"{epoch+1}/{self.__epochs}")
                    pbar.set_postfix(Message=f"Loss: {loss:.4f}")
                    #time.sleep(0.1) # Comment this for efficiency
                    pbar.update(1)
            print(f"Avg loss: {sum_loss/len(X):.4f}")
            self.__yhat = []
            
    
    def predict(self, X_test:list)->list:
        
        self.__X = X_test
        with tqdm(total=len(X_test), ncols=80) as pbar:
                for index in range(len(X_test)):
                    # Forward propagation
                    self.__forward_propagation__(index)
                    #time.sleep(0.1) # Comment this for efficiency
                    pbar.update(1)
                    self.__A1 = []
                    self.__A2 = []
                    
        return self.__yhat

    
    def predict_label(self, X_test:list)->list:
        
        self.__yhat = []
        yhat = self.predict(X_test)
        yhat_labeled = []
        
        for ypred in yhat:
            ypred_labeled = np.zeros_like(ypred)
            for word in range(self.__sz):
                max_index = np.argmax(ypred[:, word]) 
                ypred_labeled[max_index, word] = 1
            yhat_labeled.append(ypred_labeled)
            
        return yhat_labeled
        
        
# Test Model
def main():
    
    # Generate data
    X1 = np.random.randint(0, 255, (17, 20, 20))
    y1 = "There is a house."
    X2 = np.random.randint(0, 255, (5, 20, 20))
    y2 = "There is a dog."
    X3 = np.random.randint(0, 255, (10, 20, 20))
    y3 = "There is a big house"
    X4 = np.random.randint(0, 255, (7, 20, 20))
    y4 = "There is a big dog."
    X5 = np.random.randint(0, 255, (4, 20, 20))
    y5 = "There is a dog."
    X6 = np.random.randint(0, 255, (6, 20, 20))
    y6 = "There is a big dog."
    X7 = np.random.randint(0, 255, (15, 20, 20))
    y7 = "There is dog and a house."
    
    X = [X1, X2, X3, X4, X5, X6, X7]
    Y = [y1, y2, y3, y4, y5, y6, y7]
    
    # Generate filter
    fltr = np.asarray([
    [1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]
    ])
    
    # Preprocess
    X_processed = []
    for data in X:
        data_conv = np.zeros((data.shape[1]*data.shape[2], data.shape[0]))
        for t in range(data.shape[0]):
            data_conv[:, t] = Extractor(data[t, :, :], filter=fltr).flatten()
        X_processed.append(data_conv / 255)
    
    print("[X processed first point shape]")
    print(X_processed[0].shape) # First input datapoint
    
    # Encode output data
    script_size = 6
    encoder = WordEncoder(script_size)    # For RNN that return script size output, 1 words each
    #encoder = WordEncoder(script_size, 2) # For RNN that return sript size output, 2 words each
    
    #encoder.fit(Y)
    #Y_encoded = encoder.transform(Y)
    Y_encoded = encoder.fit_transform(Y)
    
    print("\n[Y encoded first point shape]")
    print(Y_encoded[0].shape)
    print("\n[Vocabulary]")
    print(encoder.vocab)
    print("\n[Vocabulary size]")
    print(encoder.vocab_size)
    
    Y_string = encoder.reverse_transform(Y_encoded)
    print(Y_string[0])
    
    # Model
    input_size = X_processed[0].shape[0]
    output_size = encoder.vocab_size
    
    descriptor = VideoToScript(input_size=input_size, output_size=output_size, n_hid_n=300, epochs=1000, script_size=script_size, lr=0.001)
    descriptor.fit(X_processed, Y_encoded)
    
    # Test
    test = [X_processed[0]]
    print("[Predict]")
    print(descriptor.predict(test))
    print("[Predict label]")
    print(descriptor.predict_label(test))
    print("[Decode]")
    print(encoder.reverse_transform(descriptor.predict_label(test)))
    
    
    
    
main()



# Notes
"""
Model:                          
                                                                          +------------------------------------------------------+  
                                                                          |  y0              y1                               yL |
                                                                          |  |               |                                |  |
                                                                          | z3,0            z3,1                             z3,L|
                                                                          |  |               |                                |  |
+-----------------------------------------------------------+             | a2,0            a2,1                             a2,L|
|                        [Encoder]                          |    [link]   |  |               |                                |  |
|a1,0 -> z1,0 -> a1,1 -> z1,1 -> a1,2 -> ... -> a1,T -> z1,T|-> a1,T+1 -> |z2,0 -> a2,0 -> z2,1 -> a2,1 -> ... -> a2,L-1 -> z2,L |
|         |               |                              |  |             |                        [Decoder]                     |
|         x0              x1                             xT |             +------------------------------------------------------+
+-----------------------------------------------------------+


Forward Propagation:
- Try using tanh instead of ReLU as activation to avoid gradient exploding.
- tanh still suffers gradient vanishing as the input of previous layer is large
even though the data is scaled. => Scale initial parameter 

- When using tanh as activation function, the activation values range from -1 to 1.
=> Wx.X should also range -1 to 1, as Wx.X scale with the sum of the number of inputsize.
- If the hidden layer also has too many neurons, then the scaled Wx would be not 
significant in comparision with the contribution of Wa1.
=> Scale all weights [Wx, Wa1] down by the total of (inputsize + neurons in hidden).

- For decoder hidden layer, the Wa2.a2 would be scaled with the number of neurons in hidden
layer (1 output scale with the number of neurons of previous output).
=> Scale Wa2 down by (neurons in hidden).

- Each output layer only scale with the output of activation at each words output.
=> No scaling needed.


Backward Propagation:
- Derivative of tanh = 1 - tanh^2
- Derivative of Li w.r.t z3i = (ypred[i] - y[i])

- Backward Propagation of output layer and hidden layer of decoder is the same.

- Backward Propagation of hidden layer of encoder is devided into 2 parts.
* The impact of changing parameters in encoder to its final output (link)
* The impact of the decoder input (link) to the final loss
- Because the multiply factor scale with the number of neurons in hidden layer.
=> scale down to avoid exploding gradient by (neurons in hidden).

Cons:
- The model must go through all the time frame in order to predict the first character.
- Gradient vanishing in encoder as the backpropagation process go back in time (multiply factor got
smaller as 1-tanh^2 and W both smaller than 1)
- The same happens to the decoder when backpropagation for parameters in encoder.
- Processing input with one hot encoder depend on the lenght of the sentence. Different sentence length
makes more 'empty' appear in a sentence.
- Due to very small gradient, the convergence occur very late.
- After 1000 epoch of SGD, the average error is 4.2267 compare to 13.1779 at the start of the trainning,
with the loss of only specific datapoint is low while other is high.
=> This maybe due to the random generated data.
"""