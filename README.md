# **Speech-Emotion-Recognition-using-Spectrograms**
Deep Learning project to classify emotions from audio spectrograms using a custom CNN model.

## Project Highlights
- CNN-based classification on 8 emotion classes.
- Custom Spectrogram Augmentation (Brightness, Contrast, Time Masking, Frequency Masking).
- Class balancing with dynamic class weights.
- Regularization: Label Smoothing, Dropout Layers.
- Learning Rate Scheduling: ReduceLROnPlateau for smart training.
- Best Model Saving automatically.
- Advanced Evaluation: Normalized Confusion Matrix, Per-class Accuracies, and Accuracy Plots.
- All plots auto-saved neatly inside the /plots folder.

## Dataset:
Images generated from speech audio using spectrograms.<br>
<br>**Image Size:** 128x128<br>
**Color:** RGB.<br>
### 8 Emotions:<br>
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised<br> 
<br>**Note: Dataset is not uploaded here due to size limitations.**


## Setup Instructions
### Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/SER-Spectrogram-Classifier.git
cd SER-Spectrogram-Classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare dataset: Place your spectrograms/ folder inside the project directory.

## Training the Model
bash
Copy
Edit
cd src
python ser_pipeline_v1.py
The script will:

Train the CNN model

Perform data augmentation dynamically

Save the best model

Create and save plots automatically in /plots

Display evaluation metrics

## Results
| Metric  | Value |
| ------------- | ------------- |
| Final Training Accuracy  | 46.70%  |
| Final Validation Accuracy  | 47.57%  |
	
## Confusion Matrix:

![confusion_matrix](https://github.com/user-attachments/assets/0a17db6f-1d00-401b-a224-1973b129cf32)


## Per-Class Validation Accuracy:

![per_class_accuracy](https://github.com/user-attachments/assets/f002f9b6-1976-4ef5-90b0-5340cb4c323d)


## Training vs Validation Accuracy:

![train_val_accuracy](https://github.com/user-attachments/assets/ee34394b-7a03-4b07-b9f7-c7cb3409310a)


## Observations
- Validation accuracy has significantly improved after using spectrogram augmentation.

- Minority classes like Happy and Sad still need better separation but improved somewhat after targeted augmentation.

- Training accuracy dropped slightly (regularization effect), but generalization is better.

- Validation accuracy even surpassed training accuracy during some epochs — a good sign against overfitting.

## Folder Structure

| Folder  | Description |
| ------------- | ------------- |
| /spectrograms  | _Input dataset (not uploaded)_  |
| /plots  | _All evaluation plots (auto-generated)_  |
| /models	| _Best saved CNN model (.keras)_ |
| /src | _Main training and augmentation scripts_ |
| /notebooks | _(Optional) EDA and visualizations_ |

## License
This project is licensed under the MIT License.

## Author
Karan Pandey

## Requirements
Basic Python packages:<br>
- tensorflow
- scikit-learn
- matplotlib
- seaborn
- numpy
