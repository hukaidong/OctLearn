import os
import zipfile
from functools import lru_cache
from os import path

import numpy

from octLearn.c.mongo_instance import MongoInstance
from octLearn.c.mongo_offline import MongoOffline
from octLearn.e.config import get_config
from octLearn.f.agent_parameters import NormalizeAgentParameters
from octLearn.f.data_rasterized import RasterizeData
from octLearn.utils import NoDataError


@lru_cache(maxsize=None)
def ObjectId2Feature(objectId: str, db_gen=None):
    configs = get_config()
    FeatRoot = configs['misc']['feat_root']
    objTail = objectId[-2:]
    dirTarget = path.join(FeatRoot, objTail)
    fileTarget = path.join(dirTarget, objectId + '.npz')

    target = None
    try:
        if os.path.exists(fileTarget):
            target = dict(numpy.load(fileTarget))
    except zipfile.BadZipfile:
        pass

    if target is None:
        print(fileTarget, 'Not Found')
        if db_gen is None:
            if configs['misc']['mongo_adapter'] == 'MongoInstance':
                MongoRecord = MongoInstance
            else:
                MongoRecord = MongoOffline
            db = MongoRecord()
        else:
            db = db_gen()

        doc = db.Case_By_id(objectId)
        if doc is None:  # BUG: Find this ghost
            print('Extract from {}: {}'.format(objectId, str(doc)[:100]), flush=True)
            raise NoDataError
        target = extract_feature(doc)
    return target


def extract_feature(document, save_result=True):
    assert document is not None
    configs = get_config()
    FeatRoot = configs['misc']['feat_root']
    rasterized = RasterizeData(document)

    objId = str(document['_id'])

    feature = dict(aid=objId, agtparm=NormalizeAgentParameters(document), cubevis=rasterized.compact_obstacle_map(),
                   taskvis=rasterized.compact_task_map(), trajvis=rasterized.compact_trajectory_map(), )

    if save_result:
        objTail = objId[-2:]
        dirTarget = path.join(FeatRoot, objTail)
        fileTarget = path.join(dirTarget, objId)

        os.makedirs(dirTarget, mode=0o755, exist_ok=True)
        numpy.savez_compressed(fileTarget, **feature)

    return feature


if __name__ == '__main__':
    from octLearn.e.config import update_config
    update_config({
        'feat_root': "/home/kaidong/normal/feature/", 
        'traj_root': '/home/kaidong/normal/trajectory/',
        'mongo_adapter': MongoInstance,
        })

    db_c = MongoInstance('normal', 'cross_valid')
    db_t = MongoInstance('normal', 'completed')

    num_processed = 0
    for caseid in db_c.Case_Ids():
        ObjectId2Feature(caseid, db_c)
        num_processed += 1
        print('processed item {}, case "completed" "{}" '.format(num_processed, caseid), end='\r')

    for caseid in db_t.Case_Ids():
        ObjectId2Feature(caseid, db_t)
        num_processed += 1
        print('processed item {}, case "cross_valid" "{}" '.format(num_processed, caseid), end='\r')
