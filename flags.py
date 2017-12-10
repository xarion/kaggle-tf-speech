import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of each training batch')
flags.DEFINE_integer('validation_batch_size', 1024, 'Size of each training batch')

# Audio, spectogram, mfcc Properties
flags.DEFINE_integer('audio_sample_rate', 16000, 'Size of each training batch')
flags.DEFINE_integer('spectogram_window_size', 480, 'Size of each training batch')
flags.DEFINE_integer('spectogram_stride', 160, 'Size of each training batch')
flags.DEFINE_integer('dtc_coefficient_count', 40, 'Size of each training batch')
flags.DEFINE_bool('mfcc_inputs', True, "Use MFCC data or raw data samples")
