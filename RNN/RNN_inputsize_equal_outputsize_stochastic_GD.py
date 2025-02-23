import numpy as np
from tqdm import tqdm
import time
import random

# Activation functions
def leaky_relu(z, alpha=0.1):
    return np.maximum(z, alpha*z)


def sigmoid(z):
    
    # Prevent sigmoid from getting to small which cause Gradient to be too large
    if z > 2.2:
        return 0.9
    if z < -2.2:
        return 0.01
    
    return 1 / (1 + np.exp(-1 * z[0, 0])) 


 # Loss function
def LossPerDataPoint(y_true, y_pred):
    
    loss = 0
    for yi, y_hati in zip(y_true, y_pred):
        if yi == 0:
            loss -= (1-yi)*np.log(1-y_hati) 
        elif yi == 1:
            loss -= yi*np.log(y_hati) 
            
    return loss / len(y_true)
    

class RNN():
        
    def __init__(self, epochs:int, n_neu_hid_l:int, input_size:int, lr:float=0.001):   
        
        # Architecture
        self.__n_neu_hid_l = n_neu_hid_l                      # number of neurons in hidden layer
        self.__epochs = epochs
        
        # Initialize parameters
        self.__W_X = np.random.rand(n_neu_hid_l, input_size)  # neurons in hidden layer
        self.__W_X /= 2*input_size # comment to not scale
        self.__W_a = np.random.rand(n_neu_hid_l, n_neu_hid_l) # ouputs from previous time activation
        self.__W_a /= 2*n_neu_hid_l # comment to not scale
        self.__W_y = np.random.rand(1, n_neu_hid_l)           # neurons in output layer
        self.__W_y /= input_size # comment to not scale
        self.__bh = np.random.rand(n_neu_hid_l, 1)            # bias in hidden layer
        self.__by = np.random.rand(1)                         # bias in output layer
        self.__lr = lr                                        # Learning rate
        
        # Get data
        self.__x_train = None
        self.__y_train = None
        
        # Archived data
        self.__A = []         # Memory for one data point, reset when next datapoint
        self.__y_hat = []     # Predicted output
        

    # Neural network flow
    def __forward_propagation_1_step__(self, index:int):
        
        size = self.__x_train[index].shape[1]
        a = np.zeros(self.__n_neu_hid_l)  # Initial activation output is 0  
        self.__A.append(a)
        y_pred = np.zeros(size)
        
        for t in range(size):
            X_t = self.__x_train[index][:, t]
            X_t = X_t[:, np.newaxis]
            z1 = np.dot(self.__W_X, X_t) + np.dot(self.__W_a, a.reshape(self.__n_neu_hid_l, 1)) + self.__bh
            #z1 = np.dot(self.__W_X, X_t) + np.dot(self.__W_a/self.__n_neu_hid_l, a.reshape(self.__n_neu_hid_l, 1)) + self.__bh # Scaled output activation weights
            a = leaky_relu(z1)
            z2 = np.dot(self.__W_y, a.reshape(self.__n_neu_hid_l, 1)) + self.__by
            y_pred[t] = sigmoid(z2)
            self.__A.append(a.reshape(self.__n_neu_hid_l))

        self.__y_hat.append(y_pred.tolist())
        

    def __backward_propagation__(self, index:int):

        y_true =  self.__y_train[index]
        y_pred = self.__y_hat[index]
        n_time = len(y_true)
        
        # Calculate gradients for W_a, W_x, bh
        GradWxL = np.zeros(self.__W_X.shape) 
        GradbhL = np.zeros(self.__bh.shape)
        GradWaL = np.zeros(self.__W_a.shape)
        
        for t in range(n_time):
            dLi_dyhi = -y_true[t]/y_pred[t] + (1-y_true[t])/(1-y_pred[t])
            dyhi_dz2 = y_pred[t]*(1-y_pred[t])
            dz2_da = self.__W_y
            da_dz1 = np.where(self.__A[t+1].reshape(self.__W_y.shape) > 0, 1, 0.1)
            
            dz1_dWx = self.__x_train[index][:, t]
            GradWxLi = np.dot((dLi_dyhi*dyhi_dz2*dz2_da*da_dz1).T, dz1_dWx[:, np.newaxis].T)
            GradWxL += GradWxLi
            
            dz1_dWa = self.__A[t]
            GradWaLi = np.dot((dLi_dyhi*dyhi_dz2*dz2_da*da_dz1).T, dz1_dWa[:, np.newaxis].T)
            GradWaL += GradWaLi
            
            dz1_dbh = 1
            GradbhLi = np.dot((dLi_dyhi*dyhi_dz2*dz2_da*da_dz1).T, dz1_dbh)
            GradbhL += GradbhLi
        
        # Update W_X    
        GradWxL /= n_time    
        self.__W_X -= self.__lr*GradWxL
        
        # Update W_a    
        GradWaL /= n_time    
        self.__W_a -= self.__lr*GradWaL
        
        # Update bh  
        GradbhL /= n_time    
        self.__bh -= self.__lr*GradbhL
        
        del GradWxL
        del GradWaL      # Will not be used after this
        del GradbhL
        
        # Calculate gradients for W_y and by
        GradWyL = np.zeros(self.__W_y.shape)
        GradbyL = 0
        
        for t in range(n_time):
            dLi_dyhi = -y_true[t]/y_pred[t] + (1-y_true[t])/(1-y_pred[t])
            dyhi_dz2 = y_pred[t]*(1-y_pred[t])
            
            dz2_dWy = self.__A[t+1].reshape(self.__W_y.shape)
            GradWyLi =  dLi_dyhi*dyhi_dz2*dz2_dWy
            GradWyL += GradWyLi
            
            dz2_dby = 1
            GraddbyLi = dLi_dyhi*dyhi_dz2*dz2_dby
            GradbyL += GraddbyLi

        # Update W_y
        GradWyL = GradWyL/n_time
        self.__W_y -= self.__lr*GradWyL 
        
        # Update by
        GradbyL = GradbyL/n_time
        self.__by -= self.__lr*GradbyL  
        
    
    def fit(self, X_train:list, y_train:list):
        
        # Assign data
        self.__x_train = X_train
        self.__y_train = y_train
        
        for epoch in range(self.__epochs):
            sum_loss = 0
            with tqdm(total=len(X_train), ncols=80) as pbar:
                for index in range(len(X_train)):
                    # Forward propagation
                    self.__forward_propagation_1_step__(index)
                    loss = LossPerDataPoint(self.__y_train[index], self.__y_hat[index])
                    sum_loss += loss
                    self.__backward_propagation__(index)
                    
                    # Format progress bar
                    pbar.set_description(f"{epoch+1}/{self.__epochs}")
                    pbar.set_postfix(Message=f"Loss: {loss:.4f}")
                    #time.sleep(0.1) # Comment this for efficiency
                    pbar.update(1)
                    self.__A = []
            print(f"Avg loss: {sum_loss/len(X_train):.4f}")
            self.__y_hat = [] # Reset for next epoch
                
    
    
    def predict(self, X_test:list):
        self.__x_train = X_test
        with tqdm(total=len(X_test), ncols=80) as pbar:
                for index in range(len(X_test)):
                    # Forward propagation
                    self.__forward_propagation_1_step__(index)
                    #time.sleep(0.1) # Comment this for efficiency
                    pbar.update(1)
                    self.__A = []
        return self.__y_hat
    
    
    def predict_label(self, X_test:list):
        label_predict = self.predict(X_test)
        
        for index in range(len(X_test)):
            label_predict[index] = np.where(np.asarray(label_predict[index]) > 0.5, 1, 0).tolist()
        
        return label_predict
       
        
# Test Model
def main():
    
    # Generate data
    # Input
    X1 = np.random.randint(0, 255 ,(3, 5))  # Time = 5, data at each time step = 3
    X2 = np.random.randint(0, 255 ,(3, 7))  # Time = 7, data at each time step = 3
    X3 = np.random.randint(0, 255 ,(3, 3))  # Time = 3, data at each time step = 3
    X4 = np.random.randint(0, 255 ,(3, 5))  # Time = 5, data at each time step = 3
    X5 = np.random.randint(0, 255 ,(3, 9))  # Time = 9, data at each time step = 3
    input_data = [X1, X2, X3, X4, X5]
    input_data_scaled = [i/100 for i in input_data ]

    # Output - two classes 0, 1
    y1 = [1, 1, 0, 1, 0]
    y2 = [0, 0, 1, 0, 1, 0, 0]
    y3 = [1, 0, 0]
    y4 = [0, 0, 1, 1, 0]
    y5 = [1, 0, 0, 0, 1, 0, 1, 1, 0]
    output_data = [y1, y2, y3, y4, y5]

    # Demo models on small data
    print("[Model 1]")
    rnn_model = RNN(epochs=5, n_neu_hid_l=2, input_size=3, lr=0.001)
    rnn_model.fit(X_train=input_data, y_train=output_data)

    print("[Model 2]")
    rnn_model2 = RNN(epochs=5, n_neu_hid_l=2, input_size=3, lr=0.001)
    rnn_model2.fit(X_train=input_data_scaled, y_train=output_data)
    
    print("[Model 3]")
    rnn_model3 = RNN(epochs=5, n_neu_hid_l=5, input_size=3, lr=0.001)
    rnn_model3.fit(X_train=input_data_scaled, y_train=output_data)
    
    # Generate large random data
    input_data = []
    output_data = []
    data_dimension_per_time = 100 # Change to modify input features vector
    n_data_points = 1000          # Change to modify number of simulated data points
    
    for _ in range(n_data_points):
        n_time_frames = random.randint(5, 30) # 5 to 30 time frames, modified if needed
        input_data.append(np.random.randint(0, 255, (data_dimension_per_time, n_time_frames)))
        output_data.append(np.random.randint(0, 2, n_time_frames).tolist())
    input_data_scaled = [i/255 for i in input_data]
    
    del rnn_model, rnn_model2, rnn_model3
    
    # Demo models on large data
    print("[Model 1]")
    rnn_model = RNN(epochs=10, n_neu_hid_l=32, input_size=100, lr=0.001)
    rnn_model.fit(X_train=input_data, y_train=output_data)

    print("[Model 2]")
    rnn_model2 = RNN(epochs=10, n_neu_hid_l=32, input_size=100, lr=1)
    rnn_model2.fit(X_train=input_data_scaled, y_train=output_data)
    
    print("[Model 3]")
    rnn_model3 = RNN(epochs=10, n_neu_hid_l=128, input_size=100, lr=0.1)
    rnn_model3.fit(X_train=input_data_scaled, y_train=output_data)
    
    # Generate test data
    test_data = []
    n_test_points = 5
    
    for _ in range(n_test_points):
        n_time_frames = random.randint(5, 30) # 5 to 30 time frames, modified if needed
        test_data.append(np.random.randint(0, 255, (data_dimension_per_time, n_time_frames)))
    test_data_scaled = [i/255 for i in input_data]
    
    # Predict
    print("[Model 1 - Predict]")
    print(rnn_model.predict(test_data)[0])
    print(rnn_model.predict_label(test_data)[0])
    
    print("[Model 2 - Predict]")
    print(rnn_model2.predict(test_data_scaled)[0])
    print(rnn_model2.predict_label(test_data_scaled)[0])
    
    print("[Model 3 - Predict]")
    print(rnn_model3.predict(test_data_scaled)[0])
    print(rnn_model3.predict_label(test_data_scaled)[0])
    
main()

# Notes
""" 
Insights:
- Problem with vanishing gradient and exploding gradient when using sigmoid and high learning rate.
- Because the previous A affect dz1_dwa which affect da_dz1 and eventually affect dL_dWA, while other factor
 only occupied 1e-2, the A increase rate is W_a^n * A (plus W_X*X also, and the data originally at 1e+2 already),
 so when the gradient is at exploding state, the model does not learn and give the same predictions and same
 gradient descent direction each time (as the simulated data is small). So a modify on the data is needed so 
 that the gradient factors can win against the growth rate of A.
- Similar problem can be witnessed with W_y and gradY.

- With large scale data and high number of neurons in hidden layer, the exploding condition occurs for the first
data point if the data is not scaled.
- Even with scaled data, the activation ouput after the hidden layer got very large quickly as the next activation
outputs are also proportioned with the number of the previous activation outputs which equal the number of neurons 
in the hidden layers => scaled down the weights of each previous activation outputs by the number of neurons.
- After scaling down the weights of W_a, the increasing speed of "a" at time t got slower but still increasing over 
time due to its proportion with weight of input layer X and as the initial weights of input are close to 1 so the
output also scale with the number of inputs. For this reason, the scaling of "a" happens while the weights are not 
updated until the model go through the whole datapoint time. => Instead of scaling down weights, scaling down the
initial weight so the activation output at time "t" is close to the activation output at time "t-1".
=> a ~ (w_X[i] * n_feature) + (w_a[i] * n_neurons * a)   (weights initial close to 1)
=> assume a = 1 unit
=> 1 ~ (w_X[i] * n_feature) + (w_a[i] * n_neurons)
=> 1 = (w_X[i]/(2*n_feature)) * n_feature + (w_a[i]/2*n_neurons)) * n_neurons
- W_y scale with GradWxLi and GradWaLi => W_y should be scale the same as W_X so the GradWxLi would be about the size
of W_X

<<<<<<< HEAD
Results of demo models on large data:
- Model 1: Even though the data is not scaled, the learning rate is low and gradient exploding does not occur.
- Model 2: The data is scaled but the learning rate is high and gradient exploding occurs occasionally and if
not, the optimization will overshoot .
- Model 3: The data is scaled, the learning rate is quite high, gradient exploding does not occur.

- Model 1 with lower number of neurons in hidden layer show better learning rate than Model 3.
- Model 3 with high number of neurons in hidden layer also show a slower learning per data point
 """
=======
- The loss does not improve: This maybe due to the randomness of the generated data and the simplicity of the model.
 """
>>>>>>> a4139887c888584e4bb1bbf675218fb36797e082
