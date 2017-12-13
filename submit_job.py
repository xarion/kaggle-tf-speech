from redis import Redis
from rq import Queue

from train_worker import run

q = Queue(connection=Redis())
from defaults import default_parameters
default_parameters.experiment_name = "test_experiment_worker"
default_parameters.validation_step = 20

q.enqueue(run, default_parameters)
