from redis import Redis
from rq import Queue

from train_worker import run

q = Queue(connection=Redis())
from defaults import default_parameters

default_parameters['model'] = "separable_resnet"
default_parameters['experiment_name'] = "separable_resnet_6"
default_parameters['mfcc_inputs'] = False
default_parameters['training'] = True
default_parameters['model_width'] = 32
default_parameters['model_setup'] = "2,2,1,1,1,2,2"
default_parameters['decay_rate'] = 0.0001
default_parameters['label_on_value'] = 0.945
default_parameters['label_off_value'] = 0.005

q.enqueue(run, default_parameters, timeout="3h")

default_parameters['training'] = False
q.enqueue(run, default_parameters, timeout="5m")


with file(default_parameters['master_folder'] + '/configurations/' + default_parameters['experiment_name'], 'w+') as f:
    f.write(str(default_parameters))
    f.flush()
    f.close()
