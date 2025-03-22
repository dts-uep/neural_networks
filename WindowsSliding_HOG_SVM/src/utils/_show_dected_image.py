import numpy as np
import cv2

def display_detected_image(image:np.array, results:list, class_list:list, box_size_ratio:list,image_name:str , save:bool=False):

    image = np.uint8(image)

    for result in results:
        box_width = result[0]
        position_x = result[1][1]
        position_y = result[1][0]
        labels = np.argwhere(result[2])
        
        for object in labels[0]:
            # Shape box for different class
            wth = int(box_width * box_size_ratio[object][0])
            lgth = int(box_width * box_size_ratio[object][1])
            
            x = int(position_x + (box_width - wth) / 2)
            y = int(position_y + (box_width - lgth) / 2)
            
            # Draw box
            image[y-8:int(y+box_width/300*40-5), x:x+wth, :] = np.array([0, 0, 255]) # up margin
            image[y+lgth-2:y+lgth, x:x+wth, :] = np.array([0, 0, 255]) # down margin
            image[y:y+lgth, x:x+2, :] = np.array([0, 0, 255]) # left margin
            image[y:y+lgth, x+wth-2:x+wth, :] = np.array([0, 0, 255]) # right margin
            
            # Draw label
            cv2.putText(image, class_list[object], (x, y+3), cv2.FONT_HERSHEY_SIMPLEX, box_width/200, (255,255,255), 1, cv2.LINE_AA)
    
    
    cv2.imshow("Detected image" ,image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    if save:
        cv2.imwrite(f"WindowsSliding_HOG_SVM/data/test_results/output_{image_name}.png", image)
        print("Saved!")
        
    