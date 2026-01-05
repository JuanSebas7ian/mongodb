import key_param
from pymongo import MongoClient
import sys

def test_connection():
    uri = key_param.mongodb_uri
    if not uri:
        print("Error: MONGODB_URI not found in environment variables.")
        print("Please ensure your .env file is set up correctly.")
        return

    print(f"Attempting to connect with URI: {uri[:15]}... (hidden)")
    
    try:
        client = MongoClient(uri)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("✅ Connection to MongoDB Atlas successful!")
        
        # Optional: Print database names to confirm permissions
        # print("Databases:", client.list_database_names())
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
