from .sift import SIFT
from .cm import ColorMoments
from .hog import HOG
from .lbp import LocalBP


def creatModel(model, **kwargs):
    model = model.lower()
    models = {"sift": SIFT, "cm": ColorMoments, "hog": HOG, "lbp": LocalBP}

    if model in models:
        return models[model](**kwargs)
    else:
        raise Exception("Supported model: sift, cm, hog, lbp")
