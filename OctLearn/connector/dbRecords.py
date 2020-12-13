import pymongo
import pymongo.database
import pymongo.collection
import bson


class MongoCollection:
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


if __name__ == '__main__':
    db = MongoCollection('learning', 'complete')
