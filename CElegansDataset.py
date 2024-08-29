import os
import json
import numpy as np
import skimage.draw
from mrcnn import utils

class CElegansDataset(utils.Dataset):
    
    def load_c_elegans(self, dataset_dir, subset):
        """Load a subset of the C. elegans dataset."""
        self.add_class("c_elegans", 1, "worm")

        # Define the path for train or validation data
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations_file = os.path.join(dataset_dir, "annotations.json")
        with open(annotations_file) as f:
            annotations = json.load(f)

        # Add images
        for image_info in annotations['images']:
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = os.path.join(dataset_dir, file_name)
            width = image_info['width']
            height = image_info['height']
            
            self.add_image(
                "c_elegans",
                image_id=image_id,
                path=image_path,
                width=width, height=height,
                annotations=annotations['annotations']
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        
        # Extract masks for each instance
        masks = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                mask = np.zeros([image_info['height'], image_info['width']], dtype=np.uint8)
                for polygon in annotation['segmentation']:
                    # Segmentation format for COCO annotations is a list of lists
                    if isinstance(polygon, list) and len(polygon) > 0:
                        # Handle multiple polygons for a single instance
                        for i in range(0, len(polygon), 2):
                            rr, cc = skimage.draw.polygon(polygon[i+1], polygon[i])
                            mask[rr, cc] = 1
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