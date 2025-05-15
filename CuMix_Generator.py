import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import tensorflow as tf

class CutMixGenerator:
    def __init__(self, base_dataset, apply_cutmix=True, alpha=1.0):
        
        self.base_dataset = base_dataset
        self.apply_cutmix_flag = apply_cutmix
        self.alpha = alpha

    def _sample_second_batch(self):

        return tf.data.experimental.get_single_element(self.base_dataset.shuffle(1000).take(1))

    def _apply_cutmix(self, data):
        
        (spectrogram1, metadata1), label1 = data
        (spectrogram2, metadata2), label2 = self._sample_second_batch()

        lam = tf.random.uniform([], 0.3, 0.7) if self.alpha == 1.0 else tf.random.beta([], self.alpha, self.alpha)

        mixed_spectrogram = lam * spectrogram1 + (1 - lam) * spectrogram2
        mixed_label = lam * label1 + (1 - lam) * label2

        return (mixed_spectrogram, metadata1), mixed_label

    def get_generator(self):

        if not self.apply_cutmix_flag:
            return self.base_dataset
        else:
            return self.base_dataset.map(lambda x, y: self._apply_cutmix((x, y)), num_parallel_calls=tf.data.AUTOTUNE)

