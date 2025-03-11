import numpy as np


# Detector




# Classifier
# Feature Engineering

# Sobel Filtering
def SobelFiltering(image_list:list, threshold:int, return_new_images:bool=False):
    
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
        

def HOG(image:list):
    
    cell_size = 8
    
    


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
