import numpy as np


def SobelFiltering(image_list:list, threshold:int=0.05, return_new_images:bool=False):
    
    im_flt_list = []
    orientation_list = []
    
    filter_x = np.asarray([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
            ])
    
    filter_y = np.asarray([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])
    
    for image in image_list:
        # X direction    
        Gx = np.zeros(image.shape)
        
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                filtered = filter_x * image[j:j+3, i:i+3]
                filtered = filtered.sum()
                Gx[j + 1, i + 1] = filtered
        Gx_denominator = np.where(Gx==0, 0.00001, Gx)
        
        # Y direction   
        Gy = np.zeros(image.shape)
         
        for i in range(image.shape[1] - 2):
            for j in range(image.shape[0] - 2):
                filtered = filter_y * image[j:j+3, i:i+3]
                filtered = filtered.sum()
                Gy[j + 1, i + 1] = filtered
         
        # Calculate magnitude
        G = np.sqrt(Gx**2 + Gy**2)
        
        if return_new_images:    
            # Convert to black white
            image_filtered = np.where(G > threshold, 255, 0)
            im_flt_list.append(image_filtered)
            
        else:
            # Calculate orentation
            im_flt_list.append(G)
            orientation = np.arctan(Gy/Gx_denominator) / np.pi * 180
            orientation = np.where(orientation < 0, 180 + orientation, orientation)
            orientation_list.append(orientation)    
    
    if return_new_images:
        return im_flt_list
    else:
        return im_flt_list, orientation_list
        

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
                histogram_vector[j] = in_bins_matrix.sum()
        
            histogram_vector_list.append(histogram_vector[:, np.newaxis])
        
        # Normalize each bloc of 2x2 celss
        n_cells_per_row = int(image.shape[1] / cell_size)
        n_cells_per_col = int(image.shape[0] / cell_size)
        feature_vector_list = []

        for i in range(n_cells_per_row - 1):
            for j in range(n_cells_per_col - 1):
                block_vector_list = [histogram_vector_list[j*n_cells_per_row+i], histogram_vector_list[j*n_cells_per_row+i+1],
                                     histogram_vector_list[(j+1)*n_cells_per_row+i], histogram_vector_list[(j+1)*n_cells_per_row+i+1]]

                # Concat into one vector
                feature_vector = np.concatenate(block_vector_list)
                feature_vector = np.clip(feature_vector / np.sqrt(np.linalg.norm(feature_vector, ord=2)**2 + 0.0001),None, 0.2)
                feature_vector = feature_vector / np.sqrt(np.linalg.norm(feature_vector, ord=2)**2 + 0.0001)
                feature_vector_list.append(feature_vector)

        data.append(np.concatenate(feature_vector_list))
        
    return data