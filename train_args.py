import tensorflow as tf

from train import Train
from defaults import default_parameters

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master_folder', default_parameters['master_folder'], 'Location to store summaries and checkpoints')
flags.DEFINE_string('model', default_parameters['model'], 'Model to use when training this')
flags.DEFINE_string('experiment_name', default_parameters['experiment_name'],
                    'Unique name to distinguish the experiment from others')

flags.DEFINE_integer('training_batch_size', default_parameters['batch_size'], 'Size of each training batch')
flags.DEFINE_integer('validation_batch_size', default_parameters['validation_batch_size'],
                     'Size of each training batch')
flags.DEFINE_integer('checkpoint_step', default_parameters['checkpoint_step'], 'Size of each training batch')
flags.DEFINE_integer('validation_step', default_parameters['validation_step'], 'Size of each training batch')

# Audio, spectogram, mfcc Properties
flags.DEFINE_integer('audio_sample_rate', default_parameters['audio_sample_rate'], 'Size of each training batch')
flags.DEFINE_integer('spectogram_window_size', default_parameters['spectogram_window_size'],
                     'Size of each training batch')
flags.DEFINE_integer('spectogram_stride', default_parameters['spectogram_stride'], 'Size of each training batch')
flags.DEFINE_integer('dtc_coefficient_count', default_parameters['dtc_coefficient_count'],
                     'Size of each training batch')
flags.DEFINE_bool('mfcc_inputs', default_parameters['mfcc_inputs'], "Use MFCC data or raw data samples")


def main(_):
    t = Train(parameters=FLAGS)
    t.train()


if __name__ == '__main__':
    tf.app.run()
