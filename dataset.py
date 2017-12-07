import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import control_flow_ops

DATA_DIR = "/home/erdi/dev/data/train/"
TRAINING_LIST = "balanced_training_list.txt"
VALIDATION_LIST = "validation_list.txt"
TEST_LIST = "test_list.txt"
SUBMISSION_LIST = "submission_list.txt"
DATASET_SPLITS = {"training": TRAINING_LIST, "validation": VALIDATION_LIST, "test": TEST_LIST,
                  "submission": SUBMISSION_LIST}
AUDIO_DIR = DATA_DIR + 'audio/'
AUDIO_SAMPLE_RATE = 16000
SPECTOGRAM_WINDOW_SIZE = 480
SPECTOGRAM_STRIDE = 160
DTC_COEFFICIENT_COUNT = 40


class Dataset:
    def __init__(self, split, batch_size):
        self.file_list = DATASET_SPLITS[split]
        self.batch_size = batch_size
        self.create_label_lookup_table()

        random_selector_variable = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
        silent_data, silent_labels = self.get_silent_records()
        labeled_data, labeled_labels = self.get_labeled_records()

        raw_data, label_id = tf.cond(tf.less(random_selector_variable, tf.constant(1. / 12.)),
                                     true_fn=lambda: (silent_data, silent_labels),
                                     false_fn=lambda: (labeled_data, labeled_labels))

        if split == "training":
            background_noise = self.get_background_noise()
        else:
            background_noise = tf.zeros([AUDIO_SAMPLE_RATE * 1, 1])

        noisy_data = raw_data + background_noise
        noisy_data = tf.clip_by_value(noisy_data, -1.0, 1.0)

        spectrogram = audio_ops.audio_spectrogram(
            noisy_data,
            window_size=SPECTOGRAM_WINDOW_SIZE,
            stride=SPECTOGRAM_STRIDE,
            magnitude_squared=True)
        mfcc = audio_ops.mfcc(
            spectrogram,
            AUDIO_SAMPLE_RATE,
            dct_coefficient_count=DTC_COEFFICIENT_COUNT)
        mfcc = tf.expand_dims(mfcc, -1)

        self.inputs, self.labels, self.background_noise, self.raw_data, self.noisy_data = \
            tf.train.shuffle_batch([mfcc, label_id, background_noise, raw_data, noisy_data],
                                   shapes=((1, 98, 40, 1), (), (16000, 1), (16000, 1), (16000, 1)),
                                   batch_size=self.batch_size,
                                   num_threads=4,
                                   capacity=batch_size * 10,
                                   min_after_dequeue=batch_size * 8)
        tf.summary.audio("raw_data", self.raw_data, sample_rate=16000)
        # tf.summary.text("label", tf.cast(self.labels, dtype=tf.string))
        tf.summary.audio("noisy_data", self.noisy_data, sample_rate=16000)

    def get_labeled_records(self):
        with tf.variable_scope("labeled_records"):
            files = self.get_file_names()
            full_file_path = self.convert_to_string_input_producer(self.get_full_path_file_names(files))

            # (AUDIO_SAMPLE_RATE * 1) because we want 1 second samples.
            full_file_name, raw_data = self.decode_wav_queue(full_file_path, AUDIO_SAMPLE_RATE * 1)

            string_parts = tf.string_split([full_file_name], '/').values
            # file_name = string_parts[-1]
            # speaker_id = tf.string_split([file_name], '_').values[0]
            label_name = string_parts[-2]
            label_id = self.label_lookup_table.lookup(label_name)

            return raw_data, label_id

    def get_silent_records(self):
        with tf.variable_scope("silent_records"):
            return self.get_background_noise(), tf.constant(self.competition_labels_to_ids["silence"])

    def get_file_names(self):
        with open(self.file_list) as f:
            files = []
            for line in f.readlines():
                files.append(line.strip())
        return files

    @staticmethod
    def get_classes_from_file_names(files):
        classes = []
        for file_name in files:
            classes.append(file_name.split('/')[0])
        return classes

    @staticmethod
    def get_full_path_file_names(files):
        return map(lambda name: AUDIO_DIR + name, files)

    @staticmethod
    def get_speaker_ids(files):
        speaker_ids = []
        for file_name in files:
            speaker_ids.append(file_name.split('/')[1].split('_')[0])
        return speaker_ids

    @staticmethod
    def decode_wav_queue(file_path, num_samples=-1):
        reader = tf.WholeFileReader()
        file_name, wav_file = reader.read(file_path)

        raw_audio, samples = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=num_samples)

        return file_name, raw_audio

    @staticmethod
    def decode_wav_file(file_path):
        wav_file = tf.read_file(file_path)
        raw_audio, samples = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=-1)
        return raw_audio

    @staticmethod
    def convert_to_string_input_producer(data):
        return tf.train.string_input_producer(tf.constant(data), shuffle=False)

    @staticmethod
    def convert_to_input_producer(data):
        return tf.train.input_producer(tf.constant(data), shuffle=False)

    @staticmethod
    def class_names_to_ids(classes):
        lookup_table, indexed_dataset = np.unique(classes, return_inverse=True)
        return indexed_dataset, lookup_table

    def get_random_background_noise(self, num_samples):
        files = [AUDIO_DIR + "_background_noise_/doing_the_dishes.wav",
                 AUDIO_DIR + "_background_noise_/dude_miaowing.wav",
                 AUDIO_DIR + "_background_noise_/exercise_bike.wav",
                 AUDIO_DIR + "_background_noise_/pink_noise.wav",
                 AUDIO_DIR + "_background_noise_/running_tap.wav",
                 AUDIO_DIR + "_background_noise_/white_noise.wav"]
        num_cases = len(files)

        background_files = map(self.decode_wav_file, files)
        # background_files = tf.convert_to_tensor(files)

        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        random_background = control_flow_ops.merge(
            [control_flow_ops.switch(background_files[case], tf.equal(sel, case))[1]
             for case in range(0, num_cases)])[0]
        random_background = tf.random_crop(random_background, [num_samples, 1])
        return random_background

    def get_background_noise(self):
        background_sample = self.get_random_background_noise(AUDIO_SAMPLE_RATE * 1)
        background_multiplier = tf.random_uniform([], minval=0, maxval=0.1, dtype=tf.float32)
        background_noise = background_sample * background_multiplier
        return background_noise

    def create_label_lookup_table(self):
        import dataset_labels

        items = dataset_labels.dataset_labels_to_competition_ids.items()

        keys, values = zip(*items)

        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
        )

        # initialize it now
        table.init.run()

        self.label_lookup_table = table
        self.competition_labels = dataset_labels.competition_labels
        self.competition_labels_to_ids = dataset_labels.competition_labels_to_ids
        self.dataset_labels = dataset_labels.dataset_labels
        self.label_lookup_dict = dataset_labels.dataset_labels_to_competition_ids
        self.number_of_labels = len(dataset_labels.competition_labels)
