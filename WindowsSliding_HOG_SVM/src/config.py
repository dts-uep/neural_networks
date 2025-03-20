# Data path
TRAIN_DATA_PATH = "WindowsSliding_HOG_SVM/data/train"
TEST_DATA_PATH = "WindowsSliding_HOG_SVM/data/test"
VAL_DATA_PATH = "WindowsSliding_HOG_SVM/data/train/validate"

# Model parameters
MODEL_INPUT_IMAGE_SIZE = (240, 240)
EPOCHS = 100000
LEARNING_RATE = 0.0001
STOP_THRESHOLD = 0

# Window sliding parameters
WINDOW_SIZE_LIST = [363, 423, 453, 501, 555, 603] # Choose size that divisible by stride_ratio as the stride chosen is the size divided by 3
ROTATION_LIST = [0] # Check only fron view
FLIP = False
STRIDE_RATIO = 3
THRESHOLD = [0.4, 0.42, 0.6, 0.6] # Based on vocab ['pedestrian', 'motorbike', 'car', 'bus']

# Draw box parameters
BOX_RATIO = [(2/3, 1), (2/3, 1), (1, 1), (1, 1)] # Based on vocab ['pedestrian', 'motorbike', 'car', 'bus']
