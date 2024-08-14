import os
from google.cloud import storage
from google.oauth2 import service_account
import uuid
def upload_test_file(bucket):
    test_file_name = f"test_file_{uuid.uuid4()}.txt"
    blob = bucket.blob(test_file_name)
    blob.upload_from_string("This is a test file.")
    return test_file_name

def test_google_cloud_access():
    if 'SERVICE_ACCOUNT_JSON_KEY' not in os.environ:
        print("Error: SERVICE_ACCOUNT_JSON_KEY not found in environment variables.")
        return False
    # Check for bucket name
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        print("Error: BUCKET_NAME not found in environment variables.")
        return False

    print("Environment variables found. Attempting to access the bucket...")
    try:
        # Initialize Google Cloud client
        credentials = service_account.Credentials.from_service_account_file(
            os.environ['SERVICE_ACCOUNT_JSON_KEY'])
        client = storage.Client(credentials=credentials)
        
        # Create a bucket object without using get_bucket
        bucket = storage.Bucket(client, bucket_name)
        
        # Upload a test file (Create operation)
        test_file_name = upload_test_file(bucket)
        print(f"Uploaded test file: {test_file_name}")
        
        # List blobs in the bucket (Read operation)
        blobs = list(client.list_blobs(bucket_name, max_results=5))
        print(f"Successfully listed contents of bucket '{bucket_name}'.")
        print("First few files in the bucket:")
        for blob in blobs:
            print(f" - {blob.name}")
        
        # Update the test file
        blob = bucket.blob(test_file_name)
        blob.upload_from_string("This is an updated test file.")
        print(f"Updated test file: {test_file_name}")
        
        # Delete the test file
        blob.delete()
        print(f"Deleted test file: {test_file_name}")
        
        return True
    except Exception as e:
        print(f"Error accessing Google Cloud Storage: {str(e)}")
        return False

if __name__ == "__main__":
    if test_google_cloud_access():
        print("Google Cloud Storage access test passed successfully!")
    else:
        print("Google Cloud Storage access test failed.")