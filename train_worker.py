from train import Train
from defaults import apply_defaults


def run(parameters):
    parameters = apply_defaults(parameters)
    t = Train(parameters)
    t.train()
