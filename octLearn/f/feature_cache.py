import os
import zipfile
from functools import lru_cache
from os import environ as ENV, path

import numpy

from octLearn.c.mongo_instance import MongoInstance
from octLearn.c.mongo_offline import MongoOffline
from octLearn.e.config import get_config
from octLearn.f.agent_parameters import NormalizeAgentParameters
from octLearn.f.data_rasterized import RasterizeData


@lru_cache(maxsize=None)
def ObjectId2Feature(objectId: str, db=None):
    FeatRoot = ENV['FeatRoot']
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
        if db is None:
            configs = get_config()
            if configs['misc']['mongo_adapter'] == 'MongoInstance':
                MongoRecord = MongoInstance
            else:
                MongoRecord = MongoOffline

            db = MongoRecord()
        doc = db.Case_By_id(objectId)
        target = extract_feature(doc)
    return target


def extract_feature(document, save_result=True):
    assert document is not None
    FeatRoot = ENV['FeatRoot']
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
