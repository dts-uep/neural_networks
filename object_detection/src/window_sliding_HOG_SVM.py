import numpy as np


# Detector




# Classifier
# Feature Engineering

# Sobel Filtering
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
        Gx = np.where(Gx==0, 0.00001, Gx)
        
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
            orientation = np.arctan(Gy/Gx) / np.pi * 180
            orientation = np.where(orientation < 0, 180 + orientation, orientation)
            orientation_list.append(orientation)    
    
    return im_flt_list if return_new_images else im_flt_list, orientation_list
        

def HOG(image_list:list)->list:
    
    cell_size = 8
    data = []
    
    for image in image_list:
        
        # Get 8x8 cells
        cells_list = []
        
        for i in range(image.shape[0] / cell_size):
            for j in range(image.shape[1] / cell_size):
                cells_list.append(image[i*8:i*8+8, j*8:j*8+8])
        
        # Get Histogram Vector
        histogram_vector_list = []
        gradient_matrix, orientation = SobelFiltering(cells_list)
        bins = (0, 20, 40, 60, 80, 100, 120, 140 , 160)
        
        for cell in cells:
            histogram_vector = np.zeros(9)
            for i in range(9):
                in_bins_matrix = np.where(orientation >= bins[i] and orientation < bins[i] + 20, cell, 0)
                histogram_vector[i] = in_bins_matrix.sum()
        
            histogram_vector_list.append(histogram_vector)
        
        # Concat into one vector
        data.append(np.concatenate(*histogram_vector_list)
        
    return data


# Flatten
def flatten_list(image_list:list)->np.array:
    return np.asarray([im.flatten() for im in image_list])



# SVM
class SVM():
    
    def __innit__(self):
        pass



# TEST
def main():
    
    # Fake data
    image_list = [np.random.randint(0, 255, (50, 50)), np.random.randint(0, 255, (50, 50))]
    print(image_list[0])
    print(SobelFiltering(image_list=image_list, threshold=0.05, return_new_images=True))
    print(SobelFiltering(image_list=image_list, threshold=0.05, return_new_images=False))

main()
