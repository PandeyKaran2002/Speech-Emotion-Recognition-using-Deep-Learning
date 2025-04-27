# Speech-Emotion-Recognition-using-Spectrograms
Deep Learning project to classify emotions from audio spectrograms using a custom CNN model.

🚀 Project Highlights
📊 CNN-based classification on 8 emotion classes.

🧪 Custom Spectrogram Augmentation (Brightness, Contrast, Time Masking, Frequency Masking).

🏷️ Class balancing with dynamic class weights.

🧹 Regularization: Label Smoothing, Dropout Layers.

📈 Learning Rate Scheduling: ReduceLROnPlateau for smart training.

💾 Best Model Saving automatically.

🔥 Advanced Evaluation: Normalized Confusion Matrix, Per-class Accuracies, and Accuracy Plots.

📸 All plots auto-saved neatly inside the /plots folder.

🗂️ Dataset
Images generated from speech audio using spectrograms.

Image Size: 128x128, Color: RGB.

8 Emotions:

Angry

Calm

Disgust

Fearful

Happy

Neutral

Sad

Surprised

Note: Dataset is not uploaded here due to size limitations.

🛠️ Setup Instructions
Clone the repo:

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

🧪 Training the Model
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

📊 Results

Metric	Value
Final Training Accuracy	46.70%
Final Validation Accuracy	47.57%
Confusion Matrix:

Per-Class Validation Accuracy:

Training vs Validation Accuracy:

📈 Observations
🎯 Validation accuracy has significantly improved after using spectrogram augmentation.

🥲 Minority classes like Happy and Sad still need better separation but improved somewhat after targeted augmentation.

📉 Training accuracy dropped slightly (regularization effect), but generalization is better.

🔥 Validation accuracy even surpassed training accuracy during some epochs — a good sign against overfitting.

📂 Folder Structure

Folder	Description
/spectrograms	Input dataset (not uploaded)
/plots	All evaluation plots (auto-generated)
/models	Best saved CNN model (.keras)
/src	Main training and augmentation scripts
/notebooks	(Optional) EDA and visualizations
🛡️ License
This project is licensed under the MIT License.

✍️ Author
Your Name

📋 Requirements
Basic Python packages:

text
Copy
Edit
tensorflow
scikit-learn
matplotlib
seaborn
numpy
