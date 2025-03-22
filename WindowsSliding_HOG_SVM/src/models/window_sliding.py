import numpy as np
import math
from src.image_processing import image_resize_equal_size
from src.feature_engineering.hog import HOG, SobelFiltering
from src import config as cf


def window_sliding_dection(image:np.array, window_size_list:list, rotation_list:list, classifier, threshold:np.array, stride_ratio:int, flip:bool=False):
    
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
    
    results = []
    
    for w in window_size_list:
        for f in flip_list:
            for a in rotation_list:
                sliding_list = []
                location_list = []
                
                for r in range(int(im_length / w * stride_ratio - stride_ratio + 1)):
                    for c in range(int(im_width / w * stride_ratio - stride_ratio + 1)):
                        
                        slide = image[int(r*w/stride_ratio):int(r*w/stride_ratio+w), int(c*w/stride_ratio):int(c*w/stride_ratio+w)]
                        
                        # Flip slide
                        f_slide = np.zeros_like(slide)
                        x = np.linspace(-(w-1)/2, (w-1)/2, w)
                        y = np.linspace(-(w-1)/2, (w-1)/2, w)
                        xv, yv = np.meshgrid(x, y)
                        
                        if f is not np.array:
                            for i in range(w-1):
                                origin_pos = np.vstack((xv[i, :], yv[i, :])).astype(int)
                                f_pos = np.dot(f, origin_pos).astype(int)
                                origin_pos = origin_pos + int((w-1)/2)
                                f_pos = f_pos + int((w-1)/2)
                                f_slide[f_pos[0, :], f_pos[1, :]] = slide[origin_pos[0, :], origin_pos[1, :]]
                        else:
                            f_slide = slide
                        
                        del slide
                         
                        # Rotate image
                        new_w = int(abs(w/2/math.cos(a-math.pi/4)*math.sqrt(2)))
                        r_slide = np.zeros((new_w,new_w))
                        
                        if a != 0.0:
                            for i in range(w-1):
                                rotation_transform = np.array([[math.cos(a), math.cos(a+math.pi/2)],
                                                                [math.sin(a), math.sin(a+math.pi/2)]])
                                origin_pos = np.vstack((xv[i], yv[i])).astype(int)
                                r_pos = np.dot(rotation_transform, origin_pos).astype(int)
                                mask = np.any(r_pos > (new_w-1)/2, axis = 0)
                                r_pos = r_pos[:,~mask]
                                origin_pos = origin_pos[:, ~mask]
                                mask = np.any(r_pos < -(new_w-1)/2, axis = 0)
                                r_pos = r_pos[:,~mask]
                                origin_pos = origin_pos[:, ~mask]
                                r_pos = r_pos + int((new_w-1)/2)
                                origin_pos = origin_pos + int((w-1)/2)
                                r_slide[r_pos[0, :], r_pos[1, :]] = f_slide[origin_pos[0, :], origin_pos[1, :]]
                        else:
                            r_slide = f_slide
                        
                        del f_slide
                        
                        sliding_list.append(r_slide)
                        location_list.append((int(r*w/stride_ratio), int(c*w/stride_ratio)))
                
                if len(sliding_list) == 0:
                    continue
                # resize
                sliding_list = image_resize_equal_size(sliding_list, cf.MODEL_INPUT_IMAGE_SIZE)
                

                # HOG + SVM
                list = HOG(sliding_list)
                class_prob_predict = classifier.predict(list)
                
                # Remove none categorized window and append in to result
                results += [[w,
                               location_list[i],
                               class_prob_predict[:, i]] for i in range(class_prob_predict.shape[1])
                               if np.any(class_prob_predict[:, i] > threshold)]
    
    print('Done')
    return results 