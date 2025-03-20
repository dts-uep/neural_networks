import numpy as np


def image_resize_equal_size(image_list:list, new_shape:tuple):

    new_image_list = []
    
    scale_r = new_shape[0] / image_list[0].shape[0] 
    scale_c = new_shape[1] / image_list[0].shape[1]
    
    for image in image_list:
        new_image = np.zeros(new_shape)
        
        for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                new_image[i, j] = image[int(i/scale_r), int(j/scale_c)]
        
        new_image_list.append(new_image)   
    
    return new_image_list


def image_resize(image_list:list, new_shape:tuple):
    
    new_image_list = []
    
    for image in image_list:
        new_image = np.zeros(new_shape)
        scale_r = new_shape[0] / image.shape[0] 
        scale_c = new_shape[1] / image.shape[1]
        
        for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                new_image[i, j] = image[int(i/scale_r), int(j/scale_c)]
        
        new_image_list.append(new_image)   
    
    return new_image_list

def color_image_resize(image_list:list, new_shape:tuple):
    
    new_image_list = []
    
    for image in image_list:
        new_image = np.zeros(new_shape)
        scale_r = new_shape[0] / image.shape[0] 
        scale_c = new_shape[1] / image.shape[1]
        
        for i in range(new_image.shape[0]):
            for j in range(new_image.shape[1]):
                new_image[i, j, :] = image[int(i/scale_r), int(j/scale_c), :]
        
        new_image_list.append(new_image)   
    
    return new_image_list


def white_padding_to_square(image_list:list):

    padded_image_list = []

    for image in image_list:
        
        if image.shape[0] >= image.shape[1]:
            square_im_size = image.shape[0]
        else:
            square_im_size = image.shape[1]

        square_im = np.ones((square_im_size, square_im_size)) * 255
        origin_pos_y = int(0 + (square_im_size - image.shape[0])/2)
        origin_pos_x = int(0 + (square_im_size - image.shape[1])/2)
        square_im[origin_pos_y:origin_pos_y+image.shape[0], origin_pos_x:origin_pos_x+image.shape[1]] = image

        padded_image_list.append(square_im)

    return padded_image_list