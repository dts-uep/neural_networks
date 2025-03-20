import numpy as np
from src.models.window_sliding import window_sliding_dection
from src.image_processing import gray_image_test, color_image_test, image_resize, color_image_resize
from src.utils import display_detected_image
import src.config as cf
import pickle as pkl
        

def main():
    
    # Get test images
    test_image = color_image_test(cf.TEST_DATA_PATH)
    gray_test_image = gray_image_test(cf.TEST_DATA_PATH)

    # Resize image for using 1 windows list
    test_image = color_image_resize(test_image, (600, 900, 3))
    gray_test_image = image_resize(gray_test_image, (600, 900))

    # Get classifier
    with open("WindowsSliding_HOG_SVM/models/svm_classifier.pkl", "rb") as file:
        svm_classifier = pkl.load(file)

    # Get encoder
    with open("WindowsSliding_HOG_SVM/models/vocab.pkl", "rb") as file:
        vocab = list(pkl.load(file).vocab.values())
    print(vocab)

    # Go through each image
    for i in range(len(test_image)):

        # Dectect object
        results = window_sliding_dection(
            gray_test_image[i],
            window_size_list=cf.WINDOW_SIZE_LIST,
            rotation_list=cf.ROTATION_LIST,
            classifier=svm_classifier,
            threshold=np.array(cf.THRESHOLD).reshape((len(vocab), 1)),
            stride_ratio=cf.STRIDE_RATIO,
            flip=cf.FLIP
            )
        
        # Draw boxes on image
        image_name = "traffic_" + str(i)

        display_detected_image(
            test_image[i],
            results=results,
            class_list=vocab,
            box_size_ratio=cf.BOX_RATIO,
            save = True,
            image_name=image_name
            )


if __name__ == "__main__":
    main()