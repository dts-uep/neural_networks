# Detect desktop

# Data path
TRAIN_DATA_PATH = "WindowsSliding_HOG_SVM/data/train"
TEST_DATA_PATH = "WindowsSliding_HOG_SVM/data/test"
VAL_DATA_PATH = "WindowsSliding_HOG_SVM/data/validate"

# Model parameters
MODEL_INPUT_IMAGE_SIZE = (280, 280)
EPOCHS = 100000
LEARNING_RATE = 0.0001
STOP_THRESHOLD = 0

# Window sliding parameters
WINDOW_SIZE_LIST = [27, 63, 81, 111, 153, 255, 273, 303, 363, 423, 453, 501] # Choose size that divisible by stride_ratio as the stride chosen is the size divided by 3
ROTATION_LIST = [0] # Check only fron view
FLIP = False
STRIDE_RATIO = 3
THRESHOLD = [4.255, 2.7, 3.97] # Based on vocab ['cup', 'lamp', 'laptop']

# Result filter parameters
RF_THRESHOLD = 0.11

# Draw box parameters
BOX_RATIO = [(1, 1), (2/3, 1), (1, 1)] # Based on vocab ['cup', 'lamp', 'laptop']


#####################################################################################
# Detect vehicles

# Data path
TRAIN_DATA_PATH2 = "WindowsSliding_HOG_SVM/data/train2"
TEST_DATA_PATH2 = "WindowsSliding_HOG_SVM/data/test2"
VAL_DATA_PATH2 = "WindowsSliding_HOG_SVM/data/validate2"

# Model parameters
MODEL_INPUT_IMAGE_SIZE2 = (280, 280)
EPOCHS2 = 100000
LEARNING_RATE2 = 0.0001
STOP_THRESHOLD2 = 0.0001

# Window sliding parameters
WINDOW_SIZE_LIST2 = [63, 81, 111, 255, 303, 363, 423, 453] # Choose size that divisible by stride_ratio as the stride chosen is the size divided by 3
ROTATION_LIST2 = [0] # Check only fron view
FLIP2 = False
STRIDE_RATIO2 = 3
THRESHOLD2 = [1.49, 4.8] # Based on vocab ['pedestrian', 'car']

# Result filter parameters
RF_THRESHOLD2 = 0.11

# Draw box parameters
BOX_RATIO2 = [(2/3, 1), (1, 1)] # Based on vocab ['pedestrian', 'car']