from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from app.config import settings


class Database:
    """
    MongoDB connection handler
    """
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    async def connect(self):
        """
        Establish MongoDB connection
        """
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[settings.DATABASE_NAME]
            # Test connection
            await self.client.admin.command('ping')
            print(f"✅ Connected to MongoDB: {settings.DATABASE_NAME}")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise e
    
    async def disconnect(self):
        """
        Close MongoDB connection
        """
        if self.client:
            self.client.close()
            print("✅ Disconnected from MongoDB")
    
    def get_collection(self, collection_name: str):
        """
        Get a MongoDB collection
        """
        if self.db is None:
            raise Exception("Database not connected")
        return self.db[collection_name]


# Global database instance
db = Database()


async def get_db():
    """
    Dependency to get database connection
    """
    return db
