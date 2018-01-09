import tensorflow as tf

from train import Train
from defaults import default_parameters

flags = tf.app.flags

flags.DEFINE_string('master_folder', default_parameters['master_folder'], 'Location to store summaries and checkpoints')
flags.DEFINE_string('model', default_parameters['model'], 'Model to use when training this')
flags.DEFINE_string('experiment_name', default_parameters['experiment_name'],
                    'Unique name to distinguish the experiment from others')

flags.DEFINE_integer('batch_size', default_parameters['batch_size'], 'Size of each training batch')
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
flags.DEFINE_bool('log_mel_inputs', default_parameters['log_mel_inputs'], "Use Log Mel Spectogram data or raw data samples")

flags.DEFINE_bool('training', default_parameters['training'],
                  "Do not set this from args. This is supposed to be handled internally.")

flags.DEFINE_string('model_setup', default_parameters['model_setup'], 'Model to use when training this')
flags.DEFINE_integer('model_width', default_parameters['model_width'], 'Model to use when training this')

flags.DEFINE_float('label_on_value', default_parameters['label_on_value'],
                   'on value for label smoothing regularization. (default) 1.0 for no smoothing.')
flags.DEFINE_float('label_off_value', default_parameters['label_off_value'],
                   'off value for label smoothing regularization. (default) 1.0 for no smoothing.')
flags.DEFINE_float('decay_rate', default_parameters['decay_rate'],
                   'Multiplier for the decay. This should increase as the number of parameters increase.')

flags.DEFINE_float('background_multiplier_min', default_parameters['background_multiplier_min'],
                   'Minimum multiplier for the background noise.')
flags.DEFINE_float('background_multiplier_max', default_parameters['background_multiplier_max'],
                   'Maximum multiplier for the background noise.')

flags.DEFINE_bool('bigram_model', default_parameters['bigram_model'],
                  "Compute 'bigrams' and reduce dimensions before the last layer.")

flags.DEFINE_integer('num_bigrams', default_parameters['num_bigrams'], 'Model to use when training this')
flags.DEFINE_bool('sigmoid_unknown', default_parameters['sigmoid_unknown'],
                     'Use a sigmoid label for unknown and silent classes')

flags.DEFINE_integer('max_model_width', default_parameters['max_model_width'], 'max number of features per layer.')

flags.DEFINE_bool('use_adam', default_parameters['use_adam'], 'Use adam optimizer instead of gradient descent')

flags.DEFINE_bool('accuracy_regulated_decay', default_parameters['accuracy_regulated_decay'],
                  'Multiplies the decay with 1 - accuracy.')

flags.DEFINE_bool('loss_regulated_decay', default_parameters['loss_regulated_decay'],
                  'Multiplies the decay with loss.')
flags.DEFINE_bool('random_resample', default_parameters['random_resample'],
                  'randomly resamples data')

flags.DEFINE_bool('global_avg_pooling', default_parameters['global_avg_pooling'],
                  "Use global average pooling in the last layer to reduce the dimensions. " +
                  "Flattens the layer if false.")

flags.DEFINE_integer('filter_size', default_parameters['filter_size'], 'length of filters(kernels) to use in convolutions')
flags.DEFINE_integer('stride_length', default_parameters['stride_length'], 'stride length to use')


flags.DEFINE_bool('class_only', default_parameters['class_only'], 'Train only softmax classes when sigmoid_unknown is true.')


FLAGS = flags.FLAGS


def main(_):
    parameters = FLAGS.__flags
    t = Train(parameters=parameters)

    t.run()


if __name__ == '__main__':
    tf.app.run()
