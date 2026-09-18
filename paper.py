"""Values fixed by the YOLOv1 paper (Redmon et al., arXiv:1506.02640)."""

INPUT_SIZE = 448 # detection runs at double the classification resolution
GRID_SIZE = 7 # S, cells per side
NUM_BOXES = 2 # B, box predictors per cell
LEAKY_SLOPE = 0.1 # negative slope of every activation but the last layer
DROPOUT = 0.5 # between the two fully connected layers

LAMBDA_COORD = 5.0 # upweights localisation
LAMBDA_NOOBJ = 0.5 # downweights confidence in cells without an object

EPOCHS = 135
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-4

HSV_FACTOR = 1.5 # exposure and saturation jitter by up to this factor

IOU_THRESHOLD = 0.5
