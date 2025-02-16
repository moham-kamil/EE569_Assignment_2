# README - Running the Model

## Requirements

Ensure you have the following installed:

- Python 3.8+
- OpenCV
- PyTorch
- Detectron2
- Matplotlib
- Numpy

## Setup

1. Clone or download the project.
2. Install required dependencies using:
3. Ensure the trained model is available at `./output/model_0009999.pth`.

## Running the Code

### Data Visualization

To display images with overlaid bounding boxes and parsing annotations, Change the path [data_root] to the actual dataset path , then run:

```bash
python 0_visualize_mhp_data.py
```

### Generate Subset from LV-MHP-v2

Change the path [LV_MHP_V2_PATH] to the actual dataset path , then run:

```bash
python 1_Generate_subset.py
```

### Mask Generation

To remove masks from face and hands, run:

```bash
python 3_Mask_generation.py
```

### Visualize Masks

To visualize the masks on the sample of images , run:

```bash
python 4_visualize_mask.py
```

This will display sample images with their annotated masks.

### Register Dataset

Run the following to register the dataset:

```bash
python 6_load_data.py
```

### Visualize Training Data

To visualize the dataset before training, run:

```bash
python 7_visualize_train_data.py
```

This will display sample images with their annotated masks and bounding boxes.

### Train the Model

Run:

```bash
python 8_train_model.py
```

### Test the Model & Visualize Predictions

Run:

```bash
python 10_test_model.py
```

This will display images with detected objects, masks, and bounding boxes.

### Evaluate Model Performance

To evaluate the trained model on the validation set, run:

```bash
python 9_evaluate_model.py
```

## Notes

- **Male** objects will be highlighted in **green**.
- **Female** objects will be highlighted in **red**.
- Ensure dataset and model paths are correct before running.


