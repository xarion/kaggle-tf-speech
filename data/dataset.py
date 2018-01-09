import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import control_flow_ops

import numpy as np

TRAINING_DATA_DIR = "/home/erdi/dev/data/train/audio/"
SUBMISSION_DATA_DIR = "/home/erdi/dev/data/test/audio/"
DATA_DIRS = {"training": TRAINING_DATA_DIR,
             "validation": TRAINING_DATA_DIR,
             "test": TRAINING_DATA_DIR,
             "submission": SUBMISSION_DATA_DIR}
TRAINING_LIST = "data/balanced_training_list.txt"

VALIDATION_LIST = "data/validation_list.txt"

TRAINING_BACKGROUND_NOISES = ["_background_noise_/doing_the_dishes.wav",
                              "_background_noise_/pink_noise.wav",
                              "_background_noise_/running_tap.wav",
                              "_background_noise_/white_noise.wav"]

VALIDATION_BACKGROUND_NOISES = ["_background_noise_/dude_miaowing.wav",
                                "_background_noise_/exercise_bike.wav"]

TEST_LIST = "data/test_list.txt"
SUBMISSION_LIST = "data/submission_list.txt"
DATASET_SPLITS = {"training": TRAINING_LIST, "validation": VALIDATION_LIST, "test": TEST_LIST,
                  "submission": SUBMISSION_LIST}


class Dataset:
    def __init__(self, split, batch_size, parameters):
        self.split = split
        self.parameters = parameters
        self.audio_dir = DATA_DIRS[self.split]
        self.file_list = DATASET_SPLITS[split]
        self.batch_size = batch_size
        self.label_lookup_table, self.dataset_labels = self.create_label_lookup_table()

        self.competition_labels = self.dataset_labels.competition_labels
        self.competition_labels_to_ids = self.dataset_labels.competition_labels_to_ids
        self.label_lookup_dict = self.dataset_labels.dataset_labels_to_competition_ids
        self.number_of_labels = len(self.dataset_labels.competition_labels)

        self.mfcc_inputs = self.parameters['mfcc_inputs']
        self.log_mel_inputs = self.parameters['log_mel_inputs']

        self.input_dimensions = (self.parameters['audio_sample_rate'], 1, 1)

        # all sets self.inputs, self.file_names, self.labels
        # The model should be careful in what it is using, because some may be placeholders.
        # Placeholders will throw an error if they need to be used out of context! so watch out for those errors

        if split == "training":
            self.input_set = self.create_training_dataset()

        elif split == "submission":
            self.input_set = self.create_submission_inputs()

        elif split == "validation":
            self.input_set = self.create_validation_inputs()

    def create_validation_inputs(self):
        with tf.device("/cpu:0"):
            random_selector_variable = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            silent_data, silent_labels = self.get_silent_records()
            labeled_data, labeled_labels = self.get_labeled_records()

            raw_data, label_id = tf.cond(tf.less(random_selector_variable, tf.constant(1. / 12.)),
                                         true_fn=lambda: (silent_data, silent_labels),
                                         false_fn=lambda: (labeled_data, labeled_labels))
            # raw_data = self.maybe_random_resample(raw_data)
            if self.mfcc_inputs:
                data = self.wav_to_mfcc(raw_data)
            elif self.log_mel_inputs:
                data = self.wav_to_log_mel_spectogram(raw_data)
            else:
                data = raw_data
                data = tf.expand_dims(data, -1)

            inputs, labels = tf.train.shuffle_batch([data, label_id],
                                                    shapes=(self.input_dimensions, ()),
                                                    batch_size=self.batch_size,
                                                    num_threads=12,
                                                    capacity=self.batch_size * 4,
                                                    min_after_dequeue=self.batch_size)
            file_names = tf.placeholder(dtype=tf.string, name="file_names_are_not_set_in_the_training_dataset")

            return inputs, labels, file_names

    def create_submission_inputs(self):
        with tf.device("/cpu:0"):
            raw_data, file_name = self.get_records(num_epochs=1)

            if self.mfcc_inputs:
                data = self.wav_to_mfcc(raw_data)
            elif self.log_mel_inputs:
                data = self.wav_to_log_mel_spectogram(raw_data)
            elif self.log_mel_inputs:
                data = self.wav_to_log_mel_spectogram(raw_data)

            else:
                data = raw_data
                data = tf.expand_dims(data, -1)

            inputs, file_names = tf.train.batch([data, file_name],
                                                shapes=(self.input_dimensions, ()),
                                                batch_size=self.batch_size,
                                                num_threads=48,
                                                capacity=self.batch_size * 10,
                                                allow_smaller_final_batch=True)
            labels = tf.placeholder(dtype=tf.int32, name="labels_are_not_set_in_the_submission_dataset")
            return inputs, labels, file_names

    def create_training_dataset(self):
        with tf.device("/cpu:0"):
            random_selector_variable = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
            silent_data, silent_labels = self.get_silent_records()
            labeled_data, labeled_labels = self.get_labeled_records()

            # labeled_data, labeled_labels = self.maybe_revert_audio_and_make_unknown(labeled_data, labeled_labels)

            raw_data, label_id = tf.cond(tf.less(random_selector_variable, tf.constant(1. / 12.)),
                                         true_fn=lambda: (silent_data, silent_labels),
                                         false_fn=lambda: (labeled_data, labeled_labels))

            raw_data = self.data_augmentation(raw_data)

            raw_data = tf.clip_by_value(raw_data, -1.2, 1.2)
            max_val = tf.reduce_max(tf.abs(raw_data))
            raw_data = tf.cond(tf.greater(max_val, 1),
                               true_fn=lambda: raw_data / max_val,
                               false_fn=lambda: raw_data)


            if self.mfcc_inputs:
                data = self.wav_to_mfcc(raw_data)
            elif self.log_mel_inputs:
                data = self.wav_to_log_mel_spectogram(raw_data)
            else:
                data = raw_data
                data = tf.expand_dims(data, -1)

            inputs, labels = tf.train.shuffle_batch([data, label_id],
                                                    shapes=(self.input_dimensions, ()),
                                                    batch_size=self.batch_size,
                                                    num_threads=32,
                                                    capacity=self.batch_size * 20,
                                                    min_after_dequeue=self.batch_size * 16)
            file_names = tf.placeholder(dtype=tf.string, name="file_names_are_not_set_in_the_training_dataset")
            return inputs, labels, file_names

    def wav_to_mfcc(self, raw_data):
        spectrogram = audio_ops.audio_spectrogram(
            raw_data,
            window_size=self.parameters['spectogram_window_size'],
            stride=self.parameters['spectogram_stride'],
            magnitude_squared=True)
        mfcc = audio_ops.mfcc(
            spectrogram,
            self.parameters['audio_sample_rate'],
            dct_coefficient_count=self.parameters['dtc_coefficient_count'])
        mfcc = tf.expand_dims(mfcc, -1)
        self.input_dimensions = (self.input_dimensions[0] / self.parameters['spectogram_stride'] - 2,
                                 self.parameters['dtc_coefficient_count'],
                                 1)
        mfcc = tf.squeeze(mfcc, 0)
        return mfcc

    def wav_to_log_mel_spectogram(self, raw_data):
        stfts = tf.contrib.signal.stft(raw_data, frame_length=1024, frame_step=512,
                                       fft_length=1024)
        magnitude_spectrograms = tf.abs(stfts)
        num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 64
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape([16000, 64, 1])
        self.input_dimensions = (16000, 64, 1)
        print mel_spectrograms.shape
        return mel_spectrograms

    def get_labeled_records(self):
        with tf.variable_scope("labeled_records"):
            files = self.get_file_names()
            full_file_path = self.convert_to_string_input_producer(self.get_full_path_file_names(files))

            # (self.parameters['audio_sample_rate'] * 1) because we want 1 second samples.
            full_file_name, raw_data = self.decode_wav_queue(full_file_path, self.parameters['audio_sample_rate'] * 1)

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

            # (self.parameters['audio_sample_rate'] * 1) because we want 1 second samples.
            full_file_name, raw_data = self.decode_wav_queue(full_file_path, self.parameters['audio_sample_rate'] * 1)

            string_parts = tf.string_split([full_file_name], '/').values
            file_name = string_parts[-1]

            return raw_data, file_name

    def get_silent_records(self):
        with tf.variable_scope("silent_records"):
            raw_data = tf.zeros([self.parameters["audio_sample_rate"], 1])
            # raw_data = self.maybe_random_resample(raw_data)
            return raw_data, tf.constant(self.competition_labels_to_ids["silence"])

    def get_file_names(self):
        with open(self.file_list) as f:
            files = []
            for line in f.readlines():
                files.append(line.strip())
            np.random.shuffle(files)
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

    def get_random_background_noise(self, with_zeros=True):

        files = TRAINING_BACKGROUND_NOISES if self.split == "training" else VALIDATION_BACKGROUND_NOISES
        files = map(lambda file_name: TRAINING_DATA_DIR + file_name, files)

        background_files = map(self.decode_wav_file, files)
        if with_zeros:
            background_files.append(tf.zeros([self.parameters['audio_sample_rate'], 1], dtype=tf.float32))
        # background_files = tf.convert_to_tensor(files)

        num_cases = len(background_files)

        sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        random_background = control_flow_ops.merge(
            [control_flow_ops.switch(background_files[case], tf.equal(sel, case))[1]
             for case in range(0, num_cases)])[0]
        random_background = tf.random_crop(random_background, [self.parameters['audio_sample_rate'], 1])
        return random_background

    def get_background_noise(self):
        background_sample = self.get_random_background_noise()
        background_multiplier = tf.random_uniform([],
                                                  minval=self.parameters['background_multiplier_min'],
                                                  maxval=self.parameters['background_multiplier_max'],
                                                  dtype=tf.float32)
        background_sample = background_sample * background_multiplier
        return background_sample

    @staticmethod
    def create_label_lookup_table():
        from data import dataset_labels

        items = dataset_labels.dataset_labels_to_competition_ids.items()
        keys, values = zip(*items)

        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values), -1
        )
        table.init.run()

        return table, dataset_labels

    def maybe_revert_audio_and_make_unknown(self, raw_data, label):
        if not True:
            unknown_label = tf.constant(self.competition_labels_to_ids["unknown"])
            random_selector_variable = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)

            return tf.cond(tf.less(random_selector_variable, tf.constant(0.95)),
                           true_fn=lambda: (raw_data, label),
                           false_fn=lambda: (tf.reverse(raw_data, axis=[0]), unknown_label))
        else:
            return raw_data, label

    def time_shift(self, data, num_samples=16000):
        def static_time_shifts(input_data, multiplier, max_samples):

            shift = num_samples / 100
            sample_length = num_samples - shift
            final_length = sample_length * multiplier
            np_indices = np.round(np.linspace(0, sample_length - 1, final_length)).astype(np.int32)
            pad_length = max_samples - len(np_indices)
            fixed_indices = tf.convert_to_tensor(np_indices)
            random_starting_point = tf.random_uniform([], minval=0, maxval=shift - 1, dtype=tf.int32)
            if multiplier == 1:
                resampled_data = input_data[random_starting_point:sample_length + random_starting_point]
            else:
                resampled_data = tf.gather(input_data[random_starting_point:], fixed_indices)
            pads = int(pad_length)
            lpad = tf.random_uniform([], minval=0, maxval=pads, dtype=tf.int32)
            rpad = pads - lpad

            return tf.pad(resampled_data, tf.convert_to_tensor([[lpad, rpad], [0, 0]]))

        multipliers = [0.8, 0.9, 1.]
        max_samples = int(max(multipliers) * num_samples)
        self.input_dimensions = (max_samples, 1, 1)
        resamplers = [static_time_shifts(data, multiplier, max_samples) for multiplier in multipliers]

        sel = tf.random_uniform([], maxval=4, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        resampled_data = control_flow_ops.merge(
            [control_flow_ops.switch(resamplers[case], tf.equal(sel, case))[1]
             for case in range(0, len(multipliers))])[0]

        return resampled_data

    def add_white_noise(self, data):
        white_noise = tf.random_uniform([self.input_dimensions[0], 1], minval=-1, maxval=1)
        return self.add_energy_scaled_noise(data, white_noise, coefficient=0.05)

    def add_background_noise(self, data):
        background_noise = self.get_random_background_noise(with_zeros=False)
        return self.add_energy_scaled_noise(data, background_noise, 0.1)

    def volume_shift(self, data):
        coefficient = tf.random_uniform([], minval=5, maxval=15, dtype=tf.float32) / 10
        return data * coefficient

    def data_augmentation(self, data):
        data = self.volume_shift(data)
        data = self.time_shift(data)
        data = self.add_white_noise(data)
        data = self.add_background_noise(data)

        return data

    def add_energy_scaled_noise(self, data, noise, coefficient):
        data_energy = tf.sqrt(tf.reduce_sum(tf.square(data)))
        noise_energy = self.get_energy(noise)
        return data + coefficient * noise * data_energy / noise_energy

    def get_energy(self, data):
        return tf.sqrt(tf.reduce_sum(tf.square(data)))