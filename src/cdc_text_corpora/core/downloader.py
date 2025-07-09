import pandas as pd
import os
from sodapy import Socrata
import pathlib
import requests
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn, DownloadColumn
from cdc_text_corpora.utils.config import get_metadata_path, get_collection_zip_path, ensure_data_directory, ensure_html_outputs_directory

_DATA_SIZE = 33567
_URL_MMWR = "https://data.cdc.gov/api/views/ut5n-bmc3/files/3c69a82b-b82e-4152-bcb9-49c3da123d1d?download=true&filename=mmwr_1982-2023.zip"
_URL_EID = "https://data.cdc.gov/api/views/ut5n-bmc3/files/3c9ce6ee-8e97-4ce4-9312-9ad3c98be408?download=true&filename=eid_1995-2023.zip"
_URL_PCD = "https://data.cdc.gov/api/views/ut5n-bmc3/files/c0594869-ba74-4c26-bf54-b2dab3dff971?download=true&filename=pcd_2004-2023.zip"

def download_file(url, _file_name):
    """Download a file with rich progress bar."""
    local_filename = _file_name
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=None,  # Use default console
        ) as progress:
            task = progress.add_task(f"Downloading {_file_name.split('/')[-1]}", total=total)
            
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        progress.update(task, advance=size)
    
    return local_filename

def download_copora(journal, url):
    """Download a journal collection to the user's current directory."""
    # Ensure html-outputs directory exists
    ensure_html_outputs_directory()
    
    # Get the full path for the collection zip file
    output_path = get_collection_zip_path(journal)
    
    download_file(url=url, _file_name=str(output_path))


def download_collection(collection: str = "all") -> None:
    """
    Download CDC Text Corpora collections. Always includes metadata.
    
    Args:
        collection: Collection to download. Options: 'pcd', 'eid', 'mmwr', 'metadata', 'all'
    """
    collection = collection.lower()
    
    collections_map = {
        'pcd': (_URL_PCD, 'pcd'),
        'eid': (_URL_EID, 'eid'), 
        'mmwr': (_URL_MMWR, 'mmwr')
    }
    
    # Always check and download metadata first (unless only downloading metadata)
    if collection != "metadata":
        metadata_path = get_metadata_path()
        if metadata_path.exists():
            print("Metadata already exists, skipping download.")
        else:
            print("Downloading metadata...")
            download_metadata()
    
    if collection == "metadata":
        metadata_path = get_metadata_path()
        if metadata_path.exists():
            print("Metadata already exists, skipping download.")
        else:
            print("Downloading metadata...")
            download_metadata()
    elif collection in collections_map:
        url, journal = collections_map[collection]
        zip_path = get_collection_zip_path(journal)
        if zip_path.exists():
            print(f"{journal.upper()} collection already exists, skipping download.")
        else:
            print(f"Downloading {journal.upper()} collection...")
            download_copora(journal=journal, url=url)
    elif collection == "all":
        print("Downloading all collections and metadata...")
        
        # Check and download each collection
        for journal, (url, name) in collections_map.items():
            zip_path = get_collection_zip_path(name)
            if zip_path.exists():
                print(f"{name.upper()} collection already exists, skipping download.")
            else:
                print(f"Downloading {name.upper()} collection...")
                download_copora(journal=name, url=url)
    else:
        available_options = list(collections_map.keys()) + ['metadata', 'all']
        raise ValueError(f"Invalid collection '{collection}'. Available options: {available_options}")


def download_metadata():
    """Download metadata to the user's current directory."""
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    client = Socrata("data.cdc.gov", None)

    # Example authenticated client (needed for non-public datasets):
    # client = Socrata(data.cdc.gov,
    #                  MyAppToken,
    #                  username="user@example.com",
    #                  password="AFakePassword")

    # First 2000 results, returned as JSON from API / converted to Python list of
    # dictionaries by sodapy.
    results = client.get("7rih-tqi5", limit=_DATA_SIZE)

    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)

    # Ensure data directory exists
    ensure_data_directory()
    
    # Get the full path for the metadata file
    output_path = get_metadata_path()
    results_df.to_csv(output_path)

if __name__ == "__main__":
    # Test the new download_collection function
    download_collection("pcd")  # Download only PCD collection