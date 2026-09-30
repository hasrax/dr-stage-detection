# Diabetic Retinopathy Stage Detection with Transfer Learning

Computer Vision coursework (BSc (Hons) Computer Science, NIBM). The notebook detects diabetic retinopathy (DR) and classifies its five ICDR stages (No DR, Mild, Moderate, Severe, Proliferative) from retinal fundus photographs, using preprocessing designed to remove camera "shortcuts", data augmentation, class balancing and a fine-tuned EfficientNetB0.

**Video demonstration:** [add YouTube link]

## Results (locked test set, 523 images)

| Metric | Value |
|---|---|
| Quadratic weighted kappa (5 stages) | 0.846 |
| Accuracy / macro-F1 (5 stages) | 74.8% / 0.578 |
| Healthy vs DR accuracy | 97.3% |
| DR sensitivity / specificity | 98.4% / 96.3% |
| DR ROC-AUC | 0.995 |
| Referable DR (Moderate or worse) sensitivity / specificity | 78.8% / 95.6% |

Validation kappa of the selected model: 0.896. On the mixed-camera "hard" subset (156 images), healthy-vs-DR accuracy stayed at 96.2%.

## Pipeline

1. **Exploratory analysis:** class imbalance and acquisition bias (image size alone predicted DR with 90.5% accuracy; field-of-view shape with 86.9%).
2. **Preprocessing:** threshold-based retina crop, square padding, common field-of-view mask, bilateral denoising, then either CLAHE and unsharp masking (v1) or Ben Graham normalisation with normalised convolution (v2). Images are processed at 512 px and resized to 224 px.
3. **Data audit and split:** 181 near-duplicates removed (37 duplicate groups had conflicting grades); stratified 70/15/15 split.
4. **Augmentation and balancing:** flips, rotation, zoom, brightness and contrast; class weights compared with oversampling.
5. **Transfer learning:** EfficientNetB0, ResNet50 and MobileNetV2 compared with two-phase fine-tuning; early stopping and learning-rate reduction on validation kappa.
6. **Evaluation:** learning curves, precision, recall, F1, confusion matrix, hard test and Grad-CAM.
7. **Prototype:** Gradio app showing the preprocessed image, stage probabilities, DR decision and Grad-CAM heatmap.

## Repository contents

| File | Description |
|---|---|
| `Computer_Vision_CW.ipynb` | Complete, commented notebook (all steps, with outputs) |
| `requirements.txt` | Python packages used |
| `results/` (optional) | Experiment log and test metrics (CSV) |
| `figures/` (optional) | Figures used in the report |

The dataset, processed images and trained model are **not** included (see below).

## How to run

1. Open the notebook in Google Colab.
2. Join the [APTOS 2019 competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection) on Kaggle, create an API token, and store it in Colab Secrets as `KAGGLE_API_TOKEN`.
3. Create a Google Drive folder named `Computer Vision CW` (or change `PROJECT_DIR` in the first cell).
4. Use a CPU runtime for data preparation and evaluation, and a T4 GPU runtime only for the training cells.
5. Run the notebook top to bottom. After the first full run, the resume cell reloads the processed data from Drive in about a minute.

All random seeds are fixed (42), and the image-processing parameters are constants in the code, so results are reproducible.

## Data and licence

The images come from the APTOS 2019 Blindness Detection dataset (Kaggle) and are subject to the competition's data rules, so they are not redistributed here. Download them with the Kaggle API as described above.

## Disclaimer

This is a coursework research prototype and not a medical device. It must not be used for diagnosis.

## Author

M. H. Gunawardena
