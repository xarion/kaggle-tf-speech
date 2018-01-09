default_parameters_dict = {
    'master_folder': '/disks/big/speech_experiments',
    'experiment_name': '',
    'batch_size': 128,
    'validation_batch_size': 512,
    'checkpoint_step': 1000,
    'validation_step': 100,
    'audio_sample_rate': 16000,
    'spectogram_window_size': 480,
    'spectogram_stride': 160,
    'dtc_coefficient_count': 40,
    'mfcc_inputs': False,
    'log_mel_inputs': False,
    'model': 'separable_resnet',
    'training': True,
    'model_setup': "1,1,1,1,1,1,1,1,1,1",
    'model_width': 8,
    'label_off_value': 0.0,
    'label_on_value': 1.0,
    'decay_rate': 0.0001,
    'bigram_model': False,
    'num_bigrams': 32,
    'global_avg_pooling': True,
    'background_multiplier_min': 0.2,
    'background_multiplier_max': 0.2,
    'filter_size': 3,
    'stride_length': 2,
    'max_model_width': 999999999999,
    'use_adam': False,
    'accuracy_regulated_decay': False,
    'loss_regulated_decay': False,
    'random_resample': True,
    'sigmoid_unknown': False,
}

default_parameters = default_parameters_dict.copy()


def apply_defaults(parameters):
    for key in default_parameters_dict.keys():
        if key not in parameters:
            parameters[key] = default_parameters_dict[key]
    return parameters
