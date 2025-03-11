import numpy as np


# Detector




# Classifier
# Feature Engineering

# Sobel Filtering
def SobelFiltering(image_list:list, threshold:int)->list:
    
    im_flt_list = []
    
    for image in image_list:
        # X direction
        filter_x = np.asarray([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
            ])
        
        Gx = np.zeros_like(image)
        
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                Gx[j + 1, i + 1] = (filter_x * image[j:j+3, i:i+3]).sum()
                
        # Y direction
        filter_y = np.asarray([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])
        
        Gy = np.zeros_like(image)
         
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                Gy[j + 1, i + 1] = (filter_y * image[j:j+3, i:i+3]).sum()
         
        # Calculate magnitude
        G = np.sqrt(Gx**2 + Gy**2)
         
        # Convert to black white
        image_filtered = np.where(G > threshold, 1, 0)
        im_flt_list.append(image_filtered)
   
    return im_flt_list
        
        
# Flatten
def flatten_list(image_list:list)->np.array:
    return np.asarray([im.flatten() for im in image_list])



# SVM
class SVM():
    
    def __innit__(self):
        pass



# TEST
def main():
    pass
    

main()
