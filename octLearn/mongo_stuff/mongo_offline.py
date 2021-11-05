import os
import pickle
from os import environ as ENV
from os.path import join as pJoin
from octLearn.e.config import get_config


class MongoOffline:
    def __init__(self, database=None, collection=None, dump_instance=None):
        configs = get_config()
        mongo_root = configs['misc']['mongo_root']
        if None in (database, collection):
            database = configs['misc']['database']
            collection = configs['misc']['collection']
        filenames = {
            'case_ids': database + '_' + collection + '_case_ids.pkl',
        }
        if dump_instance:
            os.makedirs(mongo_root, mode=0o755, exist_ok=True)
            with open(pJoin(mongo_root, filenames['case_ids']), 'wb') as f:
                pickle.dump(dump_instance.Case_Ids(), f)

        with open(pJoin(mongo_root, filenames['case_ids']), 'rb') as f:
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
    from octLearn.mongo_stuff.mongo_instance import MongoInstance
    from octLearn.e.config import update_config
    update_config({'mongo_root': '/home/kaidong/easy/database/'})
    db = MongoOffline("easy", "completed", dump_instance=MongoInstance("easy", "completed"))
    db = MongoOffline("easy", "cross_valid", dump_instance=MongoInstance("easy", "cross_valid"))
