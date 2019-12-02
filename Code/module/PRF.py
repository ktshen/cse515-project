from module.database import FilesystemDatabase
from models import modelFactory
import numpy as np

def preprocavg(v):
    return np.array(v).reshape(3, -1)[0]

def preprohog(v):   
    v = np.sum(np.array(v).reshape(-1, 9 * 4), axis = 1).reshape(19, 14) 
    return v[::2, ::2].reshape(1, -1)[0]

def prf (rel, irrel, results, table): 
# get and binarize results features
    decompMethod = 'svd'
    db = FilesystemDatabase(f"{table}_cavg", create=False)
    model = modelFactory.creatModel("cavg")
    
    cavg_th = 128
    cavg = []
    for imgId in results:
        cavg.append(preprocavg(model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)) > cavg_th)    
        
    db = FilesystemDatabase(f"{table}_hog", create=False)
    model = modelFactory.creatModel("hog")
    hog_th = 50 * 9
    hog = []
    for imgId in results:
        hog.append(preprohog(model.flattenFecture(model.deserializeFeature(db.getData(imgId)), decompMethod)) > hog_th) 
    
    feats = {}
    for idx, imgId in enumerate(results):  
        feats[imgId] = np.concatenate((cavg[idx], hog[idx]))
        
# get the sizes of both sets 
    size_R = len(rel)
    size_I = len(irrel)  

# get features for rel and irrel   
    rel_obj = []
    for imgId in rel:
        rel_obj.append(feats[imgId])
    rel_obj = np.array(rel_obj)
    
    irrel_obj = []
    for imgId in irrel:
        irrel_obj.append(feats[imgId])
    irrel_obj = np.array(irrel_obj)
    
# compute P(fi = 1(0) | rel(irrel))
    Pf1_rel = np.sum(rel_obj, axis = 0) / size_R
    Pf1_rel[Pf1_rel == 0] = 0.5 / size_R
    Pf0_rel = 1 - Pf1_rel
    
    Pf1_irrel = np.sum(irrel_obj, axis = 0) / size_I
    Pf1_irrel[Pf1_irrel == 0] = 0.5 / size_I
    Pf0_irrel = 1 - Pf1_irrel

# compute P(Oi|rel(irrel))    
    Pobj_rel = []
    Pobj_irrel = []
    
    for imgId in results:  
        Pobj_rel.append(np.prod(Pf1_rel[np.where(feats[imgId] == True)]) * np.prod(Pf0_rel[np.where(feats[imgId] == False)]))
        Pobj_irrel.append(np.prod(Pf1_irrel[np.where(feats[imgId] == True)]) * np.prod(Pf0_irrel[np.where(feats[imgId] == False)]))
    Pobj_rel = np.array(Pobj_rel)
    Pobj_rel[Pobj_rel == 0] = Pobj_rel[Pobj_rel > 0].min()
    Pobj_irrel = np.array(Pobj_irrel)
    Pobj_irrel[Pobj_irrel == 0] = Pobj_irrel[Pobj_irrel > 0].min()
# reorder results
    ratio = Pobj_rel / Pobj_irrel
    results = np.array(results)
    return list(results[ratio.argsort()[::-1]])
        

relevant = ['Hand_0000070', 'Hand_0000089', 'Hand_0000073', 'Hand_0000094', 'Hand_0000111']
irrelevant = ['Hand_0000087', 'Hand_0000112', 'Hand_0000072', 'Hand_0000074', 'Hand_0000095']
results = ['Hand_0000070', 'Hand_0000072', 'Hand_0000073', 'Hand_0000074', 'Hand_0000075'
           ,'Hand_0000087', 'Hand_0000088', 'Hand_0000089', 'Hand_0000094', 'Hand_0000095'
           , 'Hand_0000101', 'Hand_0000102', 'Hand_0000109', 'Hand_0000111', 'Hand_0000112']

r = prf(relevant, irrelevant, results, 'set2set2')
print(r)
       