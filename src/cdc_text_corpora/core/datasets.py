"""Core collections module for CDC Text Corpora.

This module provides the main collection management functionality,
including metadata operations, statistics, and collection validation.
"""

from typing import Optional, List, Dict, Any, Iterator, Union
import pathlib
import pandas as pd
from cdc_text_corpora.core.downloader import download_collection
from cdc_text_corpora.core.parser import HTMLArticleLoader, CDCArticleParser, CDCCollections, Article
from cdc_text_corpora.utils.config import get_data_directory, get_metadata_path, get_collection_zip_path


class ArticleCollection:
    """
    An iterable collection of CDC articles that uses generators for memory-efficient processing.
    
    This class reads from already parsed JSON files and yields individual Article objects
    one at a time without loading the entire collection into memory.
    """
    
    def __init__(self, corpus_manager: 'CDCCorpus', collection: Optional[str] = None, language: str = 'en') -> None:
        """
        Initialize the ArticleCollection.
        
        Args:
            corpus_manager: CDCCorpus instance
            collection: Collection name ('pcd', 'eid', 'mmwr'). If None, processes all collections
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht', or None for all)
        """
        self.corpus_manager = corpus_manager
        self.collection = collection
        self.language = language
        self._json_files = self._find_json_files()
    
    def _find_json_files(self) -> List[pathlib.Path]:
        """Find JSON files matching the collection and language criteria."""
        json_parsed_dir = self.corpus_manager.data_dir / "json-parsed"
        
        if not json_parsed_dir.exists():
            print("No json-parsed directory found. Parse some collections first.")
            return []
        
        json_files: List[pathlib.Path] = []
        
        # If specific collection requested
        if self.collection:
            if self.collection.lower() not in ['pcd', 'eid', 'mmwr']:
                raise ValueError(f"Invalid collection '{self.collection}'. Valid options: ['pcd', 'eid', 'mmwr']")
            
            # Build pattern for specific collection
            if self.language:
                pattern = f"{self.collection.lower()}_{self.language}_*.json"
            else:
                pattern = f"{self.collection.lower()}_*_*.json"
            
            json_files.extend(json_parsed_dir.glob(pattern))
        else:
            # Process all collections
            for coll in ['pcd', 'eid', 'mmwr']:
                if self.language:
                    pattern = f"{coll}_{self.language}_*.json"
                else:
                    pattern = f"{coll}_*_*.json"
                json_files.extend(json_parsed_dir.glob(pattern))
        
        return json_files
    
    def __iter__(self) -> Iterator['Article']:
        """Return iterator for the collection."""
        return self._article_generator()
    
    def _article_generator(self) -> Iterator['Article']:
        """Generator that yields individual Article objects from JSON files."""
        import json
        from cdc_text_corpora.core.parser import Article, CDCCollections
        
        for json_file in self._json_files:
            try:
                # Load JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                articles = data.get('articles', [])
                
                # Yield articles one by one
                for article_data in articles:
                    try:
                        # Convert dict back to Article object
                        article = Article(
                            title=article_data.get('title', ''),
                            abstract=article_data.get('abstract', ''),
                            full_text=article_data.get('full_text', ''),
                            references=article_data.get('references', []),
                            url=article_data.get('url', ''),
                            journal=article_data.get('journal', ''),
                            language=article_data.get('language', ''),
                            authors=article_data.get('authors', []),
                            publication_date=article_data.get('publication_date', ''),
                            collection=CDCCollections(article_data['collection']) if article_data.get('collection') else None
                        )
                        yield article
                        
                    except Exception as e:
                        print(f"Error creating Article from data: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error reading JSON file {json_file}: {e}")
                continue
    
    def __len__(self) -> int:
        """
        Get the total number of articles in the collection.
        Note: This will read JSON files to count articles.
        """
        import json
        
        total_count = 0
        for json_file in self._json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                articles = data.get('articles', [])
                total_count += len(articles)
            except Exception as e:
                print(f"Error counting articles in {json_file}: {e}")
                continue
        
        return total_count
    
    def __repr__(self) -> str:
        """String representation of the collection."""
        collection_str = self.collection or "all collections"
        language_str = self.language or "all languages"
        file_count = len(self._json_files)
        return f"ArticleCollection(collection={collection_str}, language={language_str}, json_files={file_count})"


class CDCCorpus:
    """
    Core collection manager for CDC Text Corpora.
    
    Handles collection operations including downloading, metadata management,
    statistics, and HTML article loading/parsing.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the collection manager.
        
        Args:
            data_dir: Optional path to data directory. If None, uses default location.
        """
        self.data_dir = get_data_directory(data_dir)
        self._custom_path = data_dir
    
    def download_html_articles(self, collection: str = "all") -> None:
        """
        Download CDC Text Corpora collections.
        
        Args:
            collection: Collection to download. Options: 'pcd', 'eid', 'mmwr', 'metadata', 'all'
            
        Raises:
            ValueError: If collection is not valid
        """
        valid_collections = ["pcd", "eid", "mmwr", "metadata", "all"]
        if collection.lower() not in valid_collections:
            raise ValueError(f"Invalid collection '{collection}'. Valid options: {valid_collections}")
        
        download_collection(collection.lower())
    
    def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific collection or all collections.
        
        Args:
            collection: Optional collection name. If None, returns stats for all collections.
            
        Returns:
            Dictionary containing collection statistics
        """
        stats: Dict[str, Any] = {}
        
        # Load metadata if available
        metadata_path = get_metadata_path(self._custom_path)
        if metadata_path.exists():
            try:
                df = pd.read_csv(metadata_path, low_memory=False)
                
                if collection is None:
                    # Stats for all collections
                    stats['total_documents'] = len(df)
                    stats['collections'] = df['collection'].value_counts().to_dict() if 'collection' in df.columns else {}
                    stats['available_collections'] = ['pcd', 'eid', 'mmwr']
                else:
                    # Stats for specific collection
                    if 'collection' in df.columns:
                        collection_df = df[df['collection'].str.lower() == collection.lower()]
                        stats['collection'] = collection.lower()
                        stats['document_count'] = len(collection_df)
                    else:
                        stats['collection'] = collection.lower()
                        stats['document_count'] = 0
                        stats['note'] = "Collection column not found in metadata"
                        
            except Exception as e:
                stats['error'] = f"Error reading metadata: {str(e)}"
        else:
            stats['error'] = "Metadata not found. Please run download_collection('metadata') first."
        
        return stats
    
    def list_available_collections(self) -> List[str]:
        """
        Get list of available collections.
        
        Returns:
            List of available collection names
        """
        return ['pcd', 'eid', 'mmwr']
    
    def is_collection_downloaded(self, collection: str) -> bool:
        """
        Check if a collection has been downloaded.
        
        Args:
            collection: Collection name to check
            
        Returns:
            True if collection is downloaded, False otherwise
        """
        valid_collections = ["pcd", "eid", "mmwr", "metadata"]
        if collection.lower() not in valid_collections:
            return False
        
        if collection.lower() == "metadata":
            metadata_path = get_metadata_path(self._custom_path)
            return metadata_path.exists()
        else:
            # Check for zip file
            zip_path = get_collection_zip_path(collection.lower(), self._custom_path)
            return zip_path.exists()
    
    def load_metadata(self) -> Optional[pd.DataFrame]:
        """
        Load the metadata DataFrame.
        
        Returns:
            pandas DataFrame with metadata, or None if not available
        """
        metadata_path = get_metadata_path(self._custom_path)
        if metadata_path.exists():
            try:
                return pd.read_csv(metadata_path, low_memory=False)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return None
        else:
            print("Metadata not found. Please run download_collection('metadata') first.")
            return None
    
    def load_parse_save_html_articles(self, collection: str, language: str = 'en', save_json: bool = False, output_dir: Optional[str] = None, validate_articles: bool = True) -> Dict[str, Any]:
        """
        Load and parse html articles from a collection.
        
        Args:
            collection: Collection name ('pcd', 'eid', 'mmwr')
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht', or None for all)
            save_json: Whether to save parsed articles as JSON file
            output_dir: Optional custom output directory for JSON file
            
        Returns:
            Dictionary containing loaded HTML articles, parsed articles, and optional JSON file path
        """
        if collection.lower() not in ['pcd', 'eid', 'mmwr']:
            raise ValueError(f"Invalid collection '{collection}'. Valid options: ['pcd', 'eid', 'mmwr']")
        
        # Step 1: Load HTML files
        loader = HTMLArticleLoader(collection.lower(), '', language)
        loader.load_from_file()
        
        if not loader.articles_html:
            return {
                'html_articles': {},
                'parsed_articles': {},
                'stats': {
                    'html_count': 0,
                    'parsed_count': 0,
                    'error': f'No HTML files loaded for {collection}'
                }
            }
        
        # Step 2: Parse articles with validation using collection-specific parser
        from cdc_text_corpora.core.parser import create_parser
        parser = create_parser(collection.lower(), '', language, loader.articles_html, validate_articles=validate_articles)
        parsed_articles, parsing_stats = parser.parse_all_articles()
        
        # Step 3: Save as JSON if requested
        json_file_path = None
        if save_json and parsed_articles:
            try:
                json_file_path = parser.save_as_json(parsed_articles, parsing_stats, output_dir)
            except Exception as e:
                print(f"Warning: Failed to save JSON file: {e}")
        
        # Merge parsing stats with basic stats
        combined_stats = {
            'html_count': len(loader.articles_html),
            'parsed_count': len(parsed_articles),
            'collection': collection.lower(),
            'language': language
        }
        combined_stats.update(parsing_stats)
        
        result = {
            'html_articles': loader.articles_html,
            'parsed_articles': parsed_articles,
            'stats': combined_stats
        }
        
        # Add JSON file path to result if saved
        if json_file_path:
            result['json_file_path'] = json_file_path
        
        return result
    
    def load_json_articles_as_dataframe(self, json_file_path: Optional[str] = None, collection: Optional[str] = None, language: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load parsed JSON articles as a pandas DataFrame.
        
        Args:
            json_file_path: Direct path to JSON file. If None, will look for latest file matching collection/language
            collection: Collection name to filter by (if json_file_path is None)
            language: Language to filter by (if json_file_path is None)
            
        Returns:
            pandas DataFrame with parsed articles, or None if file not found
        """
        import json
        import glob
        from datetime import datetime
        
        # If direct path provided, use it
        if json_file_path:
            target_file = pathlib.Path(json_file_path)
        else:
            # Search for JSON files in json-parsed directory
            json_parsed_dir = self.data_dir / "json-parsed"
            
            if not json_parsed_dir.exists():
                print("No json-parsed directory found. Parse some collections first.")
                return None
            
            # Build search pattern
            if collection and language:
                pattern = f"{collection}_{language}_*.json"
            elif collection:
                pattern = f"{collection}_*_*.json"
            elif language:
                pattern = f"*_{language}_*.json"
            else:
                pattern = "*.json"
            
            # Find matching files
            json_files = list(json_parsed_dir.glob(pattern))
            
            if not json_files:
                print(f"No JSON files found matching pattern: {pattern}")
                return None
            
            # Use the most recent file
            target_file = max(json_files, key=lambda f: f.stat().st_mtime)
            print(f"Loading most recent file: {target_file}")
        
        # Load and parse JSON file
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract articles list
            articles = data.get('articles', [])
            
            if not articles:
                print(f"No articles found in {target_file}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            # Add metadata columns
            df['loaded_from'] = str(target_file)
            df['loaded_at'] = datetime.now().isoformat()
            
            print(f"✅ Loaded {len(df)} articles from {target_file}")
            return df
            
        except Exception as e:
            print(f"Error loading JSON file {target_file}: {e}")
            return None
    
    def load_json_articles_as_iterable(self, collection: Optional[str] = None, language: str = 'en') -> ArticleCollection:
        """
        Load articles as an iterable ArticleCollection for memory-efficient processing.
        
        Args:
            collection: Collection name ('pcd', 'eid', 'mmwr'). If None, processes all collections
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht', or None for all)
            
        Returns:
            ArticleCollection: An iterable collection of articles
        """
        return ArticleCollection(self, collection, language)
    
    def get_data_directory(self) -> pathlib.Path:
        """
        Get the data directory path.
        
        Returns:
            Path to the data directory
        """
        return self.data_dir
    
    def __repr__(self) -> str:
        """String representation of the collection manager."""
        return f"CDCCorpus(data_dir='{self.data_dir}')"