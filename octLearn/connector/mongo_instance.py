import bson
import pymongo
import pymongo.collection
import pymongo.database

from octLearn.e.config import get_config


class MongoInstance:
    client: pymongo.MongoClient
    db: pymongo.database.Database
    col: pymongo.collection.Collection

    def __init__(self, database=None, collection=None):
        self.client = pymongo.MongoClient()
        if None in (database, collection):
            configs = get_config()
            database = configs['misc']['database']
            collection = configs['misc']['collection']
        self.db = self.client[database]
        self.col = self.db[collection]

    def Case_Ids(self):
        return [str(x['_id']) for x in self.col.find({}, {'_id': 1})]

    def Case_By_id(self, doc_id: str):
        objId = bson.ObjectId(doc_id)
        return self.col.find_one({'_id': objId})

    def queue_agent_request(self, agent_parameters):
        parameter_list = []
        for i in range(agent_parameters.shape[0]):
            parameter_list.append({'agent id': i, 'agent parameters': agent_parameters[i].tolist()})

        document = {'agent parameters': parameter_list}
        self.db['queued'].insert_one(document)
