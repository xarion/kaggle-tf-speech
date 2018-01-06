def get_model(params):
    if params["model"] == "separable_resnet":
        from separable_resnet import separable_resnet
        return separable_resnet.Model
    else:
        from mfcc_simple import model
        return model.Model