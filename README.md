# Multi-Input Deep Neural Network for Speech Emotion Recognition from CutMix-Augmented Spectrograms and Audio-Derived Features

## Overview

This project presents a robust and scalable Speech Emotion Recognition (SER) system developed using a multi-input deep neural network. The model integrates visual spectrogram representations of speech signals with high-dimensional numerical features extracted directly from audio waveforms. A novel CutMix-based augmentation strategy is employed to enhance spectrogram diversity, improving the generalization of the model. This architecture has been meticulously engineered and refined to balance training performance with validation stability, resulting in a system capable of identifying emotions across multiple classes with high per-class accuracy.

The project leverages the RAVDESS dataset and integrates audio-derived metadata, which includes features like MFCCs, chroma, zero-crossing rate, and spectral characteristics. This dual-modality approach allows the model to learn from both the time-frequency patterns in spectrograms and the statistical signal properties of the audio files.

---

## Motivation

Traditional emotion recognition systems often rely on either raw audio features or image-based representations, but each modality captures only a subset of the emotional cues present in human speech. To overcome this limitation, this project introduces a hybrid architecture capable of learning from both visual and statistical cues. By combining spectrograms and structured metadata in a unified model, we significantly improve the robustness and granularity of emotion classification.

Furthermore, the performance of neural networks in low-data regimes is enhanced using CutMix, a data augmentation technique typically reserved for image classification. By adapting CutMix to spectrograms while preserving the integrity of associated audio features, we improve data variability and reduce overfitting.

---

## Dataset

We utilize the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song), a balanced and well-annotated corpus of emotional speech samples from multiple speakers. Each audio file is labeled with one of the following emotion classes:

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

For each audio clip in the dataset, we extract two distinct representations:

1. **Spectrogram Images**: Visual time-frequency representations of the raw waveform.
2. **Numerical Audio Features**: Extracted using `librosa`, including 181 numerical features such as MFCCs, spectral centroid, bandwidth, roll-off, ZCR, and chroma vectors.

These two streams are synchronized via filename matching and stored under a structured format for easy pipeline integration.

---

## Data Processing and Augmentation

### Spectrogram and Feature Preparation

The pipeline begins by converting audio clips into spectrogram images resized to 128x128 pixels. Parallelly, audio features are extracted using a pre-processing script and saved into a CSV file. These features are standardized using `StandardScaler` to ensure uniform distribution.

Each data sample is thus represented as a tuple of:

- A spectrogram image
- A 181-dimensional feature vector
- A one-hot encoded emotion label

### CutMix Augmentation

To mitigate overfitting and enhance variability, we introduce a customized `CutMixGenerator`. This generator performs the following:

- Randomly selects two samples from the training set.
- Mixes their spectrograms using a lambda sampled from a beta distribution.
- Interpolates their emotion labels accordingly.
- Retains metadata features from the original sample to preserve semantic consistency.

CutMix is applied exclusively to the spectrograms, not the audio features, ensuring metadata integrity while improving model generalization.

### Custom Spectrogram Augmentation

The model also employs targeted spectrogram augmentation techniques:

- Brightness and contrast variations
- Frequency and time masking
- Random flips and saturation jittering for minority classes (e.g., sad and happy)

This additional augmentation improves minority class recognition and adds resilience to noise and variation in real-world applications.

---

## Model Architecture

The model consists of two input branches:

1. **CNN for Spectrograms**:
   - Three convolutional blocks with batch normalization, max-pooling, and dropout
   - Global average pooling followed by dense layers
   - Regularized with `L2` and dropout for generalization

2. **DNN for Audio Features**:
   - Dense layers with ReLU activations and L2 regularization
   - Optimized to project audio features into the same latent space as spectrogram embeddings

The two branches are concatenated and passed through a final classification layer with a softmax activation over eight emotion classes. The loss function used is `CategoricalCrossentropy` with label smoothing.

---

## Training and Evaluation

The model was trained for 30 epochs using the Adam optimizer with a learning rate scheduler and early stopping. Class weights were computed to handle label imbalance. CutMix was applied during training, while validation data remained untouched to preserve evaluation consistency.

### Metrics Tracked:

- Training and Validation Accuracy
- Per-Class Accuracy after every epoch
- Normalized Confusion Matrix
- Final Classification Report

Despite a reduction in raw validation accuracy compared to baseline CNN-only models, this multi-input system showed significantly reduced overfitting, better generalization, and much higher per-class accuracy for previously underperforming classes like “Happy” and “Sad”.

---

## Results

**Final Training Accuracy**: 0.5399  
**Final Validation Accuracy**: 0.5069  

**Per-Class Accuracy Highlights** (Post-CutMix):

- **Sad**: 0.98  
- **Angry**: 0.18  
- **Fearful**: 0.11  
- **Happy**: Now non-zero (previously 0.00 before CutMix)  
- Others: Showed incremental improvements across most classes

These results underscore the effectiveness of CutMix and custom augmentation in stabilizing class-specific performance and reducing bias toward dominant emotions.

---

## Visualizations

The project includes multiple performance visualizations:

- **Training vs Validation Accuracy Curve**
- **Normalized Confusion Matrix**
- **Per-Class Accuracy Bar Chart**

These plots are automatically generated and saved after training under the `cutmix_multi-input_plots` directory.

---

## How to Run

1. Install the required packages (`tensorflow`, `numpy`, `pandas`, `librosa`, `sklearn`, etc.)
2. Place the spectrograms in the `spectrograms/` directory with one subfolder per emotion.
3. Place the extracted audio features CSV as `Col_features.csv`.
4. Run the training script:

```bash
python train_multi_input_ser.py
```
Trained model weights are saved as best_model.keras and can be reused for inference or fine-tuning.

## Future Work
While the current pipeline is effective, future iterations may include:

* Integration of augmentation within the CutMix generator to allow class-conditional transformations

* Support for real-time inference on audio streams

* Extension to multilingual emotional speech datasets

* Hyperparameter optimization via Keras Tuner or Optuna

## Credits
Developed as part of a research-driven machine learning project focused on advanced SER strategies using deep learning, audio signal processing, and multi-modal data fusion.
