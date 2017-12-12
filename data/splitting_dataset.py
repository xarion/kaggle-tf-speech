from dataset import Dataset
import tensorflow as tf


# This may extend the Dataset itself for some functionality. it's fine.
class SplittingDataset():

    def __init__(self, training_batch_size, validation_batch_size):
        training_dataset = Dataset(split="training", batch_size=training_batch_size)
        validation_dataset = Dataset(split="validation", batch_size=validation_batch_size)
        self.do_validate = tf.placeholder_with_default(False, ())
        self.inputs, self.labels, self.file_names = tf.cond(self.do_validate,
                                                            lambda: validation_dataset.input_set,
                                                            lambda: training_dataset.input_set)
        self.number_of_labels = training_dataset.number_of_labels
