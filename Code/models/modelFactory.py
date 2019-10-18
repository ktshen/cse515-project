from .sift import SIFT
from .cm import ColorMoments
from .hog import HOG
from .lbp import LocalBP

MODELS = {"sift": SIFT, "cm": ColorMoments, "hog": HOG, "lbp": LocalBP}


def getSupportModel():
    return list(MODELS.keys())


def creatModel(model, **kwargs):
    model = model.lower()

    if model in MODELS:
        return MODELS[model](**kwargs)
    else:
        raise Exception("Supported model: sift, cm, hog, lbp")
