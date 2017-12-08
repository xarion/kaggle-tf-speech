import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import control_flow_ops

TRAINING_DATA_DIR = "/home/erdi/dev/data/train/audio/"
SUBMISSION_DATA_DIR = "/home/erdi/dev/data/test/audio/"
DATA_DIRS = {"training": TRAINING_DATA_DIR,
             "validation": TRAINING_DATA_DIR,
             "test": TRAINING_DATA_DIR,
             "submission": SUBMISSION_DATA_DIR}
TRAINING_LIST = "balanced_training_list.txt"
VALIDATION_LIST = "validation_list.txt"
TEST_LIST = "test_list.txt"
SUBMISSION_LIST = "submission_list.txt"
DATASET_SPLITS = {"training": TRAINING_LIST, "validation": VALIDATION_LIST, "test": TEST_LIST,
                  "submission": SUBMISSION_LIST}

AUDIO_SAMPLE_RATE = 16000
SPECTOGRAM_WINDOW_SIZE = 480
SPECTOGRAM_STRIDE = 160
DTC_COEFFICIENT_COUNT = 40


class Dataset:
    def __init__(self, split, batch_size):
        self.split = split
        self.audio_dir = DATA_DIRS[self.split]
        self.file_list = DATASET_SPLITS[split]
        self.batch_size = batch_size
        self.create_label_lookup_table()

        # all sets self.inputs, self.file_names, self.labels
        # The model should be careful in what it is using, because some may be placeholders.
        # Placeholders will throw an error if they need to be used out of context! so watch out for those errors

        if split == "training":
            with tf.device("/cpu:0"):
                random_selector_variable = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
                silent_data, silent_labels = self.get_silent_records()
                labeled_data, labeled_labels = self.get_labeled_records()

                raw_data, label_id = tf.cond(tf.less(random_selector_variable, tf.constant(1. / 12.)),
                                             true_fn=lambda: (silent_data, silent_labels),
                                             false_fn=lambda: (labeled_data, labeled_labels))

                background_noise = self.get_background_noise()
                noisy_data = raw_data + background_noise
                noisy_data = tf.clip_by_value(noisy_data, -1.0, 1.0)

            mfcc = self.wav_to_mfcc(noisy_data)
            self.inputs, self.labels = tf.train.shuffle_batch([mfcc, label_id],
                                                              shapes=((1, 98, 40, 1), ()),
                                                              batch_size=self.batch_size,
                                                              num_threads=32,
                                                              capacity=batch_size * 20,
                                                              min_after_dequeue=batch_size * 16)
            self.file_names = tf.placeholder(dtype=tf.string, name="file_names_are_not_set_in_the_training_dataset")

        elif split == "submission":
            with tf.device("/cpu:0"):
                raw_data, file_name = self.get_records(num_epochs=1)

            mfcc = self.wav_to_mfcc(raw_data)
            self.inputs, self.file_names = tf.train.batch([mfcc, file_name],
                                                          shapes=((1, 98, 40, 1), ()),
                                                          batch_size=self.batch_size,
                                                          num_threads=48,
                                                          capacity=batch_size * 10,
                                                          allow_smaller_final_batch=True)
            self.labels = tf.placeholder(dtype=tf.int32, name="labels_are_not_set_in_the_submission_dataset")
        elif split == "validation":
            with tf.device("/cpu:0"):
                raw_data, label_id = self.get_labeled_records()
                mfcc = self.wav_to_mfcc(raw_data)
                self.inputs, self.labels = tf.train.shuffle_batch([mfcc, label_id],
                                                                  shapes=((1, 98, 40, 1), ()),
                                                                  batch_size=self.batch_size,
                                                                  num_threads=4,
                                                                  capacity=batch_size * 2,
                                                                  min_after_dequeue=batch_size)
                self.file_names = tf.placeholder(dtype=tf.string, name="file_names_are_not_set_in_the_training_dataset")

    @staticmethod
    def wav_to_mfcc(raw_data):
        spectrogram = audio_ops.audio_spectrogram(
            raw_data,
            window_size=SPECTOGRAM_WINDOW_SIZE,
            stride=SPECTOGRAM_STRIDE,
            magnitude_squared=True)
        mfcc = audio_ops.mfcc(
            spectrogram,
            AUDIO_SAMPLE_RATE,
            dct_coefficient_count=DTC_COEFFICIENT_COUNT)
        mfcc = tf.expand_dims(mfcc, -1)
        return mfcc

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

    def get_records(self, num_epochs=None):
        with tf.variable_scope("records"):
            files = self.get_file_names()
            full_file_path = self.convert_to_string_input_producer(self.get_full_path_file_names(files), num_epochs)

            # (AUDIO_SAMPLE_RATE * 1) because we want 1 second samples.
            full_file_name, raw_data = self.decode_wav_queue(full_file_path, AUDIO_SAMPLE_RATE * 1)

            string_parts = tf.string_split([full_file_name], '/').values
            file_name = string_parts[-1]

            return raw_data, file_name

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

    def get_full_path_file_names(self, files):
        return map(lambda name: self.audio_dir + name, files)

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
    def convert_to_string_input_producer(data, num_epochs=None):
        return tf.train.string_input_producer(tf.constant(data), shuffle=False, num_epochs=num_epochs)

    @staticmethod
    def convert_to_input_producer(data):
        return tf.train.input_producer(tf.constant(data), shuffle=False)

    @staticmethod
    def class_names_to_ids(classes):
        lookup_table, indexed_dataset = np.unique(classes, return_inverse=True)
        return indexed_dataset, lookup_table

    def get_random_background_noise(self, num_samples):

        files = [TRAINING_DATA_DIR + "_background_noise_/doing_the_dishes.wav",
                 TRAINING_DATA_DIR + "_background_noise_/dude_miaowing.wav",
                 TRAINING_DATA_DIR + "_background_noise_/exercise_bike.wav",
                 TRAINING_DATA_DIR + "_background_noise_/pink_noise.wav",
                 TRAINING_DATA_DIR + "_background_noise_/running_tap.wav",
                 TRAINING_DATA_DIR + "_background_noise_/white_noise.wav"]
        num_cases = len(files)

        background_files = map(self.decode_wav_file, files)
        background_files.append(tf.zeros([num_samples, 1], dtype=tf.float32))
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
