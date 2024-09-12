# Instance Segmentation of C. Elegans

This repository contains the implementation and code for the instance segmentation of *C. elegans* using Mask R-CNN and a combination of Mask R-CNN with U-Net. The project was carried out in the context of a computer vision course, and the results and methodology are explained in detail in the report. The primary goal is to accurately segment *C. elegans* in image data, focusing on the potential of Mask R-CNN and the integration of U-Net to enhance segmentation performance.

## Project Structure

- **`data_prep.ipynb`**: This notebook contains the data preprocessing pipeline, which organizes the dataset in the required folder structure, loads the images, and processes the annotations. It also includes the visualization of some sample images and their corresponding segmentation masks and bounding boxes.
  
- **`inspect_data.ipynb`**: This notebook is used for inspecting the dataset, visualizing individual images, and checking for any anomalies in the annotations. It provides further insights into the dataset's structure and quality.

- **`train_maskRcnn_colab.ipynb`**: This notebook implements the training of the Mask R-CNN model using Google Colab's TPU. It includes the configuration setup, the training process with COCO pre-trained weights, and evaluation of the model's performance.

- **`train_maskRcnn_uNet_colab.ipynb`**: This notebook implements the combined Mask R-CNN with U-Net as the backbone. Although the model was successfully built, it encountered a shape mismatch error during training. The details of the architecture and the challenges faced during training are discussed in the report.
  
## Running the Notebooks

- Start by running the `data_prep.ipynb` notebook to prepare the dataset for training.
- Use `inspect_data.ipynb` to explore and visualize the data.
- For training Mask R-CNN, run `train_maskRcnn_colab.ipynb` in Google Colab.
- To experiment with the combined U-Net and Mask R-CNN architecture, run `train_maskRcnn_uNet_colab.ipynb` in Google Colab.


