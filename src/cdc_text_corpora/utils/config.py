"""Configuration utilities for cdc-text-corpora package."""

import os
import pathlib
from typing import Optional


def get_data_directory(custom_path: Optional[str] = None) -> pathlib.Path:
    """
    Get the data directory path where CDC corpus data should be stored.
    
    This function determines the appropriate directory for storing CDC corpus data
    based on the user's current working directory when they run the package.
    
    Args:
        custom_path: Optional custom path to use instead of default
        
    Returns:
        Path object pointing to the data directory
        
    Notes:
        - If custom_path is provided, uses that path
        - Otherwise, creates 'cdc-corpus-data' in the current working directory
        - This ensures data is stored where the user is running the package from
    """
    if custom_path is not None:
        return pathlib.Path(custom_path)
    
    # Use current working directory where the user is running the package
    user_cwd = pathlib.Path.cwd()
    data_dir = user_cwd / "cdc-corpus-data"
    
    return data_dir


def get_metadata_path(custom_path: Optional[str] = None) -> pathlib.Path:
    """
    Get the full path to the metadata CSV file.
    
    Args:
        custom_path: Optional custom data directory path
        
    Returns:
        Path object pointing to the metadata CSV file
    """
    data_dir = get_data_directory(custom_path)
    return data_dir / "cdc_corpus_df.csv"


def get_collection_zip_path(collection: str, custom_path: Optional[str] = None) -> pathlib.Path:
    """
    Get the full path to a collection's ZIP file.
    
    Args:
        collection: Name of the collection (pcd, eid, mmwr)
        custom_path: Optional custom data directory path
        
    Returns:
        Path object pointing to the collection ZIP file
    """
    data_dir = get_data_directory(custom_path)
    return data_dir / "html-outputs" / f"{collection.lower()}.zip"


def ensure_data_directory(custom_path: Optional[str] = None) -> pathlib.Path:
    """
    Ensure the data directory exists, creating it if necessary.
    
    Args:
        custom_path: Optional custom data directory path
        
    Returns:
        Path object pointing to the created data directory
    """
    data_dir = get_data_directory(custom_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_html_outputs_directory(custom_path: Optional[str] = None) -> pathlib.Path:
    """
    Ensure the html-outputs directory exists, creating it if necessary.
    
    Args:
        custom_path: Optional custom data directory path
        
    Returns:
        Path object pointing to the created html-outputs directory
    """
    data_dir = get_data_directory(custom_path)
    html_outputs_dir = data_dir / "html-outputs"
    html_outputs_dir.mkdir(parents=True, exist_ok=True)
    return html_outputs_dir