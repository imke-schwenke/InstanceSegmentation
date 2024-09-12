import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# root directory of the project
#ROOT_DIR = '/content/gdrive/MyDrive'
ROOT_DIR = './'

# import mask rcnn
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CElegansConfig(Config):
    NAME = "c_elegans"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + worm
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################  

class CElegansDataset(utils.Dataset):

    def load_c_elegans(self, dataset_dir, subset):
        """Load a subset of the C. elegans dataset."""
        self.add_class("c_elegans", 1, "c_elegans")
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations_file = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        images = annotations_file['images']

        for image_info in images:
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = os.path.join(dataset_dir, file_name)
            width = image_info['width']
            height = image_info['height']

            filtered_annotations = [
                annotation for annotation in annotations_file['annotations']
                if annotation['image_id'] == image_id
            ]

            self.add_image(
                "c_elegans",
                image_id=image_id,
                path=image_path,
                width=width, height=height,
                annotations=filtered_annotations
            )
    

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        #image_info = self.image_info[image_id]
        image_info = self.image_info[image_id-1]
        annotations = image_info['annotations']
        
        # Extract masks for each instance
        masks = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                mask = np.zeros([image_info['height'], image_info['width']], dtype=np.uint8)
                for polygon in annotation['segmentation']:
                    # Segmentation format for COCO annotations is a list of lists
                    if isinstance(polygon, list) and len(polygon) > 0:
                        # Extrahiere die x- und y-Koordinaten
                        x_coords = polygon[0::2]  # Jeder zweite Wert, startend bei index 0 (x-Koordinaten)
                        y_coords = polygon[1::2]  # Jeder zweite Wert, startend bei index 1 (y-Koordinaten)

                        # Überprüfe, ob die Koordinaten innerhalb der Bildgrenzen liegen
                        if (min(x_coords) >= 0 and max(x_coords) < image_info['width'] and 
                            min(y_coords) >= 0 and max(y_coords) < image_info['height']):
                            # Zeichne das Polygon
                            rr, cc = skimage.draw.polygon(y_coords, x_coords)

                            # Setze die Maske
                            mask[rr, cc] = 1
                            # # Handle multiple polygons for a single instance
                            # for i in range(0, len(polygon), 2):
                            #     rr, cc = skimage.draw.polygon(polygon[i+1], polygon[i])
                            #     mask[rr, cc] = 1
                masks.append(mask)
        
        if len(masks) == 0:
            return np.zeros([image_info['height'], image_info['width'], 0]), np.zeros([0], dtype=np.int32)
        
        masks = np.stack(masks, axis=-1)
        class_ids = np.ones([masks.shape[-1]], dtype=np.int32)  # Assuming only one class
        
        return masks, class_ids
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]