import numpy as np
from src.models.window_sliding import window_sliding_dection
from src.image_processing import gray_image_test, color_image_test
from src.utils import display_detected_image, result_filter
import src.config as cf
import pickle as pkl
        

# Detect desktop
def main():
    
    # Get test images
    test_image = color_image_test(cf.TEST_DATA_PATH)
    gray_test_image = gray_image_test(cf.TEST_DATA_PATH)

    # Get classifier
    with open("WindowsSliding_HOG_SVM/models/svm_classifier.pkl", "rb") as file:
        svm_classifier = pkl.load(file)

    # Get encoder
    with open("WindowsSliding_HOG_SVM/models/encoder.pkl", "rb") as file:
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
        
        # Filter boxes
        results = result_filter(results=results, class_list=vocab, class_threshold=cf.THRESHOLD, inter_threshold=cf.RF_THRESHOLD)

        # Draw boxes on image
        image_name = "test_result_" + str(i)

        display_detected_image(
            test_image[i],
            results=results,
            class_list=vocab,
            box_size_ratio=cf.BOX_RATIO,
            save = True,
            image_name=image_name
            )
        

# Detect vehicles
def main2():
    
    # Get test images
    test_image = color_image_test(cf.TEST_DATA_PATH2)
    gray_test_image = gray_image_test(cf.TEST_DATA_PATH2)

    # Get classifier
    with open("WindowsSliding_HOG_SVM/models/svm_classifier2.pkl", "rb") as file:
        svm_classifier = pkl.load(file)

    # Get encoder
    with open("WindowsSliding_HOG_SVM/models/encoder2.pkl", "rb") as file:
        vocab = list(pkl.load(file).vocab.values())
    print(vocab)

    # Go through each image
    for i in range(len(test_image)):

        # Dectect object
        results = window_sliding_dection(
            gray_test_image[i],
            window_size_list=cf.WINDOW_SIZE_LIST2,
            rotation_list=cf.ROTATION_LIST2,
            classifier=svm_classifier,
            threshold=np.array(cf.THRESHOLD2).reshape((len(vocab), 1)),
            stride_ratio=cf.STRIDE_RATIO2,
            flip=cf.FLIP2
            )
        
        # Filter boxes
        results = result_filter(results=results, class_list=vocab, class_threshold=cf.THRESHOLD2, inter_threshold=cf.RF_THRESHOLD2)

        # Draw boxes on image
        image_name = "test_result_2" + str(i)

        display_detected_image(
            test_image[i],
            results=results,
            class_list=vocab,
            box_size_ratio=cf.BOX_RATIO2,
            save = True,
            image_name=image_name
            )


if __name__ == "__main__":
    #main()
    main2()