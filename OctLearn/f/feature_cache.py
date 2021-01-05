import os
import zipfile
from functools import lru_cache
from os import environ as ENV, path

import numpy

from octLearn.connector.dbRecords import MongoInstance
from octLearn.f.data_rasterized import RasterizeData


@lru_cache(maxsize=100)
def ObjectId2Feature(objectId: str):
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
        mongo = MongoInstance('learning', 'completed')
        doc = mongo.Case_By_id(objectId)
        target = extract_feature(doc)
    return target


def GetAgentParameters(document):
    agent_parameters = document['agent parameters']
    params = numpy.zeros([len(agent_parameters), len(agent_parameters[0]['agent parameters'])])
    parameter_means = numpy.array([1.2, 8.0, 90.0, 0.75])
    for d in agent_parameters:
        aid, par = d.values()
        params[aid, :] = (par / parameter_means) - 1
    return params


def extract_feature(document, save_result=True):
    FeatRoot = ENV['FeatRoot']
    rasterized = RasterizeData(document)

    objId = str(document['_id'])

    feature = dict(aid=objId, agtparm=GetAgentParameters(document), cubevis=rasterized.compact_obstacle_map(),
                   taskvis=rasterized.compact_task_map(), trajvis=rasterized.compact_trajectory_map(), )

    if save_result:
        objTail = objId[-2:]
        dirTarget = path.join(FeatRoot, objTail)
        fileTarget = path.join(dirTarget, objId)

        os.makedirs(dirTarget, mode=0o755, exist_ok=True)
        numpy.savez_compressed(fileTarget, **feature)

    return feature
