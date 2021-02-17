import os
import pickle
from os import environ as ENV
from os.path import join as pJoin


class MongoOffline:
    def __init__(self, database, collection, dump_instance=None):
        root = ENV['MongoRoot']
        filenames = {
            'case_ids': database + '_' + collection + '_case_ids.pkl',
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

    def Case_By_id(self, obj_id):
        raise NotImplementedError('MongoOffline#Case_By_id: Full mongodb indexing not available.')

    def find(self, *args, **kwargs):
        raise NotImplementedError()

if __name__ == "__main__":
    from octLearn.connector.mongo_instance import MongoInstance
    db = MongoOffline("easy", "completed", dump_instance=MongoInstance("easy", "completed"))