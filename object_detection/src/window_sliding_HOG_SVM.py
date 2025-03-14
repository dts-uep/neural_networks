import numpy as np
import math


# One hot encoder
class OneHotEncoder():
    
    def __innit__(self):
        
        self.__class_list = None
        self.__index_list = None
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


# Image resizing
def square_image_resize(image:list, new_size):
    pass



# Feature Engineering
def SobelFiltering(image_list:list, threshold:int=0.05, return_new_images:bool=False):
    
    im_flt_list = []
    orientation_list = []
    
    for image in image_list:
        # X direction
        filter_x = np.asarray([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
            ])
        
        Gx = np.zeros(image.shape)
        
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                filtered = filter_x * image[j:j+3, i:i+3]
                filtered = filtered.sum()
                Gx[j + 1, i + 1] = filtered / 510.0 # Scale down 510, Max = 1020, Min = -1020
        Gx_denominator = np.where(Gx==0, 0.00001, Gx)
        
        # Y direction
        filter_y = np.asarray([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])
        
        Gy = np.zeros(image.shape)
         
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                filtered = filter_y * image[j:j+3, i:i+3]
                filtered = filtered.sum()
                Gy[j + 1, i + 1] = filtered / 510.0
         
        # Calculate magnitude
        G = np.sqrt(Gx**2 + Gy**2)
        
        if return_new_images:    
            # Convert to black white
            image_filtered = np.where(G > threshold, 1, 0)
            im_flt_list.append(image_filtered)
            
        else:
            # Calculate orentation
            im_flt_list.append(G)
            orientation = np.arctan(Gy/Gx_denominator) / np.pi * 180
            orientation = np.where(orientation < 0, 180 + orientation, orientation)
            orientation_list.append(orientation)    
    
    return im_flt_list if return_new_images else im_flt_list, orientation_list
        

def HOG(image_list:list)->list:
    
    cell_size = 8
    data = []
    
    for image in image_list:
        
        # Get 8x8 cells
        cells_list = []
        
        for i in range(int(image.shape[0] / cell_size)):
            for j in range(int(image.shape[1] / cell_size)):
                cells_list.append(image[i*8:i*8+8, j*8:j*8+8])
        
        # Get Histogram Vector
        histogram_vector_list = []
        gradient_matrix, orientation = SobelFiltering(cells_list)
        bins = (0, 20, 40, 60, 80, 100, 120, 140 , 160)
        
        for i in range(len(cells_list)):
            histogram_vector = np.zeros(9)
            for j in range(9):
                in_bins_matrix = np.where(orientation[i] >= bins[j], gradient_matrix[i], 0)
                in_bins_matrix = np.where(orientation[i] < bins[j] + 20, in_bins_matrix, 0)
                histogram_vector[j] = in_bins_matrix.sum() / cell_size**2  # Scale by number of pixel per cell
        
            histogram_vector_list.append(histogram_vector[:, np.newaxis])
        
        # Concat into one vector
        data.append(np.concatenate(histogram_vector_list))
        
    return data


def flatten_list(image_list:list)->np.array:
    return np.asarray([im.flatten() for im in image_list])


# Classifier - SVM
class SVM():
    
    def __init__(self, n_class:int, input_size:int):
        
        self.__W = np.random.rand(n_class, input_size+1) / input_size
        self.__X = None
        self.__Y = None
        self.__Z = None
        self.__loss = 0
        self.__c = n_class
        self.__lr = None
    
    
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
        self.__W -= self.__lr*gradient_matrix / self.__X.shape[1]
        
    
    def fit(self, X:list, Y:list, epochs:int=1, lr:float=0.001):
        
        # Set up
        self.__X = np.hstack(X)
        self.__X = np.vstack([self.__X, np.ones((1, len(X)))])
        self.__lr = lr
        
        self.__Y = np.hstack(Y)
        
        for _ in range(epochs):
            
            self.__forward__()
            self.__update_weight__()
            
            print(f"Loss: {self.__loss}")
     

    def predict(self, X:list):
        
        self.__X = np.hstack(X)
        self.__X = np.vstack([self.__X, np.ones((1, len(X)))])
        self.__forward__()
        
        return self.__Z
        
    
    def predict_label(self, X:list):
        
        y_label = []
        y_pred = self.predict(X)
        
        for n in range(len(X)):
            highest_score = np.argmax(y_pred[:, n])
            label = np.zeros((self.__c, 1))
            label[highest_score] = 1
            y_label.append(label)
        
        return y_label


# Detector          
def object_detect(image:np.array, window_size_list:list, rotation_list:list, classifier, image_size:int, flip:bool=False):
    
    # Image properties
    im_width = image.shape[1]
    im_length = image.shape[0]
    
    # Initiate transform matrices
    classifier = classifier
    
    # Windows flip vertically or not
    if flip:
        flip_list = [1, np.asarray([[1, 0], [0, -1]])]
    else:
        flip_list = [1]
    
    # Windows rotation of alpha to check object rotation of negative alpha
    rotation_list = [math.radians(alpha) for alpha in rotation_list]
    
    result = []
    
    for w in window_size_list:
        for f in flip_list:
            for a in rotation_list:
                attribute = [w, f, rot]
                sliding_list = []
                location_list = []
                label_list = []
                
                for r in range(im_length / w * 3 - 2):
                    for c in range(im_width / w * 3 - 2):
                        
                        slide = image[r*w/3:r*w/3+w, c*w/3:c*w/3+w]
                        
                        # Flip slide
                        f_slide = np.zeros_like(slide)
                        x = np.linspace(0, w, w)
                        y = np.linspace(0, w, w)
                        xv, yv = np.meshgrid(x, y)
                        
                        for i in range(w):
                            origin_pos = np.vstack(xv, yv)
                            f_pos = np.dot(f, origin_pos)
                            f_slide[f_pos] = slide[origin_pos]
                            
                        # Rotate image
                        new_w = int(abs(w/2/math.cos(a-math.pi/4)*math.sqrt(2)))
                        r_slide = np.zeros((new_w,new_w))
                        
                        for i in range(w):
                            origin_pos = np.vstack(xv, yv)
                            r_pos = np.dot(r, origin_pos).astype(int)
                            mask = np.any(r_pos > new_w, axis = 0)
                            r_pos = r_pos[:,~mask]
                            r_slide[r_pos] = slide[origin_pos]
                        
                        sliding_list.append(r_slide)
                        location_list.append((r*w/3, c*w/3))
                
                # resize
                            
                        
                        
                        
        


# TEST
def main():
    
    # Fake data
    image_list = [np.random.randint(0, 255, (80, 80)), np.random.randint(0, 255, (80, 80)), np.random.randint(0, 255, (80, 80)), np.random.randint(0, 255, (80, 80)), np.random.randint(0, 255, (80, 80))]
    print(image_list[0])
    print(SobelFiltering(image_list=image_list, threshold=0.05, return_new_images=True))
    print(SobelFiltering(image_list=image_list, threshold=0.05, return_new_images=False))
    
    print(HOG(image_list))
    Y = ['car', 'bike', 'pedestrian', 'bus', 'bus']
    encoder = OneHotEncoder()
    print(encoder.fit_transform(Y))
    print(encoder.reverse_transform(encoder.fit_transform(Y)))
    
    svm = SVM(encoder.vocab_size, 900)
    svm.fit(HOG(image_list), encoder.transform(Y), epochs=10000)
    
    image_list_test = [np.random.randint(0, 255, (80, 80))]
    print(svm.predict(HOG(image_list=image_list_test)))
    print(svm.predict_label(HOG(image_list=image_list_test)))
    print(encoder.reverse_transform(svm.predict_label(HOG(image_list=image_list_test))))
    print(encoder.vocab)
    
    print(encoder.reverse_transform(svm.predict_label(HOG(image_list=image_list[0:1]))))

#main()


def test():
    
    object_detect([], [], [0, 45, 60], None, 80, False)
    object_detect([], [], [0, -45, -60], None, 80, True)

test()