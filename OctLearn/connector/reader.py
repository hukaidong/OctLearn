import pymongo
import pymongo.database
import pymongo.collection
import bson


class MongoCollection():
    client: pymongo.MongoClient
    db: pymongo.database.Database
    col: pymongo.collection.Collection

    def __init__(self, database, collection):
        self.client = pymongo.MongoClient()
        self.db = self.client[database]
        self.col = self.db[collection]

    def Case_By_id(self, doc_id: str):
        objId = bson.ObjectId(doc_id)
        return self.col.find_one({'_id': objId})
