import numpy as np
import cv2
import os

# Get data
def gray_image_data(folder_path:str):
    
    image_list = []
    label_list = []
    
    class_list = os.listdir(folder_path)
    for class_ in class_list:
        images_files = os.listdir(folder_path + "/" + class_)
        image_list += [cv2.imread(folder_path + "/" + class_ + "/" + im, 0) for im in images_files]
        label_list += [class_] * len(images_files)

    return image_list, label_list


def color_image_data(folder_path:str):
    
    image_list = []
    label_list = []
    
    class_list = os.listdir(folder_path)
    for class_ in class_list:
        images_files = os.listdir(folder_path + "/" + class_)
        image_list += [cv2.imread(folder_path + "/" + class_ + "/" + im, 1) for im in images_files]
        label_list += [class_] * len(images_files)

    return image_list, label_list

def color_image_test(folder_path:str):
    
    image_list = []
    image_list += [cv2.imread(folder_path  + "/" + im, 1) for im in os.listdir(folder_path)]

    return image_list

def gray_image_test(folder_path:str):
    
    image_list = []
    image_list += [cv2.imread(folder_path  + "/" + im, 0) for im in os.listdir(folder_path)]

    return image_list

