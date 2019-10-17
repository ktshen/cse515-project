from module.database import FilesystemDatabase
from module.DimRed import DimRed
from models import modelFactory
import argparse

parser = argparse.ArgumentParser(description="Phase 2 Task 1")
parser.add_argument(
    "-m",
    "--model",
    metavar="model",
    type=str,
    help="The model will be used.",
    required=True,
)
parser.add_argument(
    "-t",
    "--table",
    metavar="table",
    type=str,
    help="The table will be used.",
    required=True,
)
parser.add_argument(
    "-k", 
    "--topk", 
    metavar="topk", 
    type=int, 
    help="K.", 
    required=True
)
parser.add_argument(
    "-d",
    "--method",
    metavar="method",
    type=str,
    help="The method will be used to reduce dimension.",
    required=True,
)
args = parser.parse_args()

# extract argument
model = args.model.lower()
table = args.table.lower()
topk = args.topk
decompMethod = args.method.lower()

# Create database according to model and table name
db = FilesystemDatabase(f"{table}_{model}", create=False)
# Removed unused variable in case misusing.
del table

# Load features of images
model = modelFactory.creatModel(model)
objFeat = []

for keyId in db.keys():
    objFeat.append(model.flattenFecture(model.deserializeFeature(db.getData(keyId)), model))

latentModel = DimRed.createReduction(decompMethod, k=topk, data = objFeat)
latentModel.printLatentSemantics()