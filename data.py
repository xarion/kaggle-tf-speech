import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import audio_ops

DATA_DIR = "/home/erdi/dev/data/train/"
TRAINING_LIST = "training_list.txt"
AUDIO_DIR = DATA_DIR + 'audio/'
AUDIO_SAMPLE_RATE = 16000
SPECTOGRAM_WINDOW_SIZE = 30
SPECTOGRAM_STRIDE = 10
DTC_COEFFICIENT_COUNT = 40


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size


    @staticmethod
    def get_file_names():
        with open(TRAINING_LIST) as f:
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
    def decode_wav(file_path):
        reader = tf.WholeFileReader()
        key, wav_file = reader.read(file_path)
        raw_audio, samples = audio_ops.decode_wav(wav_file, desired_channels=1, desired_samples=AUDIO_SAMPLE_RATE)
        raw_audio = tf.expand_dims(raw_audio, dim=-1)
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

    def create_spectogram_from_files(self, files):

        set_size = len(files)
        string_labels = self.get_classes_from_file_names(files)
        int_labels, self.label_lookup_table = self.class_names_to_ids(string_labels)
        labels = self.convert_to_string_input_producer(int_labels)

        full_file_path = self.convert_to_string_input_producer(self.get_full_path_file_names(files))
        input_id = self.convert_to_input_producer(range(0, set_size))
        speaker_id, spea = self.convert_to_string_input_producer(self.get_speaker_ids(files))

        raw_wav_data = self.decode_wav(full_file_path)

        spectrogram = audio_ops.audio_spectrogram(
            raw_wav_data,
            window_size=SPECTOGRAM_WINDOW_SIZE,
            stride=SPECTOGRAM_STRIDE,
            magnitude_squared=True)
        self.mfcc = audio_ops.mfcc(
            spectrogram,
            AUDIO_SAMPLE_RATE,
            dct_coefficient_count=DTC_COEFFICIENT_COUNT)

        self.raw_wav_data_queue = tf.train.input_producer(raw_wav_data, shuffle=False)

        self.inputs, self.input_ids, self.labels, self.speaker_ids = tf.train.shuffle_batch(
            [self.raw_wav_data_queue.dequeue(), input_id.dequeue(), labels.dequeue(), speaker_id.dequeue()],
            shapes=((98, 40, 1), (), (), ()),
            batch_size=self.batch_size,
            min_after_dequeue=self.batch_size * 4,
            num_threads=6,
            capacity=self.dataset_size)

    def get_traning_samples(self):
