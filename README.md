# **Speech-Emotion-Recognition-using-Spectrograms**
Deep Learning project to classify emotions from audio spectrograms using a custom CNN model.

## About the Project

This project focuses on **Speech Emotion Recognition (SER)** — the task of automatically detecting the emotional state of a speaker based on their speech.

Instead of using raw audio signals directly, the audio was first converted into **spectrograms** — visual representations of audio frequencies over time.  
Spectrograms allow convolutional neural networks (CNNs) to learn meaningful spatial patterns from audio, treating the problem like an image classification task.

The goal was to train a custom CNN model that could classify speech into one of **eight emotion categories**:  
**Angry**, **Calm**, **Disgust**, **Fearful**, **Happy**, **Neutral**, **Sad**, and **Surprised**.

This project bridges the fields of **speech processing** and **computer vision**, leveraging visual deep learning techniques to understand audio emotions.

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

---

## Challenges Faced

- **Limited Dataset Size:**  
  Unlike large datasets like ImageNet, speech emotion datasets are relatively small, making deep learning harder without overfitting.

- **Highly Imbalanced Classes:**  
  Some emotions like *Happy* and *Sad* had fewer examples compared to others, requiring special handling to prevent model bias.

- **Spectrogram Complexity:**  
  Emotional cues in speech are often subtle and difficult to visualize, especially in overlapping frequency regions.

- **Model Overfitting:**  
  Early experiments showed the model memorizing training data but failing on unseen validation samples, leading to the need for techniques like **Dropout**, **Label Smoothing**, and **Aggressive Data Augmentation**.

- **Noise in Audio Samples:**  
  Variations and background noise in some speech recordings introduced unwanted patterns into the spectrograms.

---

## Applications

Speech Emotion Recognition systems have wide-ranging applications, such as:

- **Virtual Assistants (Alexa, Siri, Google Assistant):**  
  Enabling emotional awareness in smart home devices.

- **Mental Health Monitoring:**  
  Detecting early signs of stress, depression, or anxiety from a patient's speech.

- **Call Center Analytics:**  
  Monitoring customer satisfaction and frustration automatically in customer service centers.

- **Human-Robot Interaction:**  
  Equipping robots with the ability to recognize and respond to human emotions.

- **E-Learning Systems:**  
  Assessing student engagement and emotional state during remote learning sessions.

- **Driver Safety Systems:**  
  Monitoring driver anger, fatigue, or drowsiness through their voice commands.

- **Interactive Gaming:**  
  Creating emotionally adaptive gameplay experiences based on the player’s voice.

---

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
