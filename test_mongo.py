from services.db import db

print("Connected successfully!")
print("DB:", db.name)
print("Collections:", db.list_collection_names())

from services.db import analytics_collection

analytics_collection.insert_one({
    "event": "mongo_connected",
    "status": "ok"
})