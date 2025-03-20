import numpy as np

def gray_scale(image:np.array):
    return 0.2989*image[:, :, 2]+0.5870*image[:, :, 1]+0.1140*image[:, :, 0]

def Laplacian_image_sharpen(image:np.array):

    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

    laplacian_image = np.zeros_like(image)

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            laplacian_image[i, j] = (image[i-1:i+2, j-1:j+2]*laplacian_kernel).sum()
    
    return image - laplacian_image