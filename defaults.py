default_parameters_dict = {
    'master_folder': '/disks/big/speech_experiments',
    'experiment_name': '',
    'batch_size': 128,
    'validation_batch_size': 1024,
    'checkpoint_step': 1000,
    'validation_step': 100,
    'audio_sample_rate': 16000,
    'spectogram_window_size': 480,
    'spectogram_stride': 160,
    'dtc_coefficient_count': 40,
    'mfcc_inputs': True,
    'model': 'default_mfcc_model'
}

default_parameters = default_parameters_dict.copy()


def apply_defaults(parameters):
    for key in default_parameters_dict.keys():
        if key not in parameters:
            parameters[key] = default_parameters_dict[key]
    return parameters
