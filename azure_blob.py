from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os

load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTIONSTRING")
CONTAINER_NAME = "keywords"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def check_files_in_blob(file_name):
    """
    Check if the file exists in Azure Blob Storage.
    """
    try:
        blob_list = [blob.name for blob in container_client.list_blobs()]
        return file_name in blob_list
    except Exception as e:
        print(f"An error occurred while checking blob storage: {e}")
        return False

def save_data_to_blob(file_name, data):
    """
    Save data to Azure Blob Storage as a .txt file.
    """
    try:
        blob_client = container_client.get_blob_client(file_name)
        # Save the data as a text file
        blob_client.upload_blob(data, overwrite=True)
        print(f"Data saved to Azure Blob Storage: {file_name}")
    except Exception as e:
        print(f"An error occurred while saving to blob storage: {e}")

def fetch_data_from_blob(file_name):
    """
    Fetch data from Azure Blob Storage.
    """
    try:
        blob_client = container_client.get_blob_client(file_name)
        downloaded_blob = blob_client.download_blob().readall()
        return downloaded_blob.decode('utf-8')  # Convert bytes to string
    except Exception as e:
        print(f"An error occurred while fetching from blob storage: {e}")
        return None
