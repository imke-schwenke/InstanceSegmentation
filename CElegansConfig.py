from mrcnn.config import Config

class CElegansConfig(Config):
    NAME = "c_elegans"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + worm
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9