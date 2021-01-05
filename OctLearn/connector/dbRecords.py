import os
import pickle
from os import environ as ENV
from os.path import join as pJoin

import bson
import pymongo
import pymongo.collection
import pymongo.database


class MongoInstance:
    client: pymongo.MongoClient
    db: pymongo.database.Database
    col: pymongo.collection.Collection

    def __init__(self, database, collection):
        self.client = pymongo.MongoClient()
        self.db = self.client[database]
        self.col = self.db[collection]

    def Case_Ids(self):
        return [str(x['_id']) for x in self.col.find({}, {'_id': 1})]

    def Case_Random_N(self, N):
        cursor = self.col.aggregate([{'$sample': {'size': N}}])
        return [*cursor]

    def Case_By_id(self, doc_id: str):
        objId = bson.ObjectId(doc_id)
        return self.col.find_one({'_id': objId})

    def find(self, *args, **kwargs):
        return self.col.find(*args, **kwargs)


class MongoOffline:
    def __init__(self, database, collection, dump_instance=None):
        root = ENV['MongoRoot']
        filenames = {
            'case_ids': database + '_' + collection + '_case_ids.pkl'
        }
        if dump_instance:
            os.makedirs(root, mode=0o755, exist_ok=True)
            with open(pJoin(root, filenames['case_ids']), 'wb') as f:
                pickle.dump(dump_instance.Case_Ids(), f)

        with open(pJoin(root, filenames['case_ids']), 'rb') as f:
            self._case_ids = pickle.load(f)

    def Case_Ids(self):
        return self._case_ids

    def Case_Random_N(self, *args, **kwargs):
        raise NotImplementedError()

    def Case_By_id(self, *args, **kwargs):
        raise NotImplementedError()

    def find(self, *args, **kwargs):
        raise NotImplementedError()


if __name__ == '__main__':
    db = MongoInstance('learning', 'complete')
