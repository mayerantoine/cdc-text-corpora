#!/usr/bin/env python3
"""Test script for CDC batch article parser."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cdc_text_corpora.core.parser import HTMLArticleLoader, CDCArticleParser, Article, CDCCollections
from cdc_text_corpora.utils.config import get_collection_zip_path
from pathlib import Path

def test_batch_parsing():
    """Test batch parsing of articles."""
    
    # Check if downloaded zip files exist
    collections = ['pcd', 'eid', 'mmwr']
    
    for collection in collections:
        zip_path = get_collection_zip_path(collection)
        print(f"\nChecking {collection.upper()} collection at: {zip_path}")
        
        if not zip_path.exists():
            print(f"❌ {collection.upper()} zip file not found. Please download first with: cdc-corpus download -c {collection}")
            continue
            
        print(f"✅ {collection.upper()} zip file found")
        
        # Test loading and parsing articles
        try:
            print(f"\n{'='*60}")
            print(f"TESTING BATCH PARSING FOR {collection.upper()}")
            print('='*60)
            
            # Step 1: Load HTML files
            print("Step 1: Loading HTML files...")
            loader = HTMLArticleLoader(collection, '', 'en')
            loader.load_from_file()
            
            if not loader.articles_html:
                print(f"❌ No HTML files loaded for {collection}")
                continue
            
            print(f"✅ Loaded {len(loader.articles_html)} HTML files")
            
            # Step 2: Initialize parser with loaded HTML
            print("Step 2: Initializing parser...")
            parser = CDCArticleParser(collection, '', 'en', loader.articles_html)
            
            # Step 3: Parse all articles
            print("Step 3: Parsing articles...")
            articles = parser.parse_all_articles()
            
            if articles:
                print(f"\n✅ Successfully parsed {len(articles)} articles")
                
                # Show some sample results
                sample_articles = list(articles.items())[:3]
                for i, (url, article) in enumerate(sample_articles):
                    print(f"\nSample Article {i+1}:")
                    print(f"  URL: {url}")
                    print(f"  Title: {article.title}")
                    print(f"  Authors: {len(article.authors)} found")
                    print(f"  Abstract: {'✓' if article.abstract else '✗'}")
                    print(f"  Full Text: {len(article.full_text)} chars")
                    print(f"  References: {len(article.references)} found")
                    
            else:
                print(f"❌ No articles parsed for {collection}")
                
        except Exception as e:
            print(f"❌ Error during batch parsing: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    test_batch_parsing()