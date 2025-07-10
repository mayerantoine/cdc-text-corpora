#!/usr/bin/env python3
"""Test script for CDC article parser."""

import sys
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cdc_text_corpora.core.parser import CDCArticleParser, Article, CDCCollections
from pathlib import Path

def test_single_article_parsing():
    """Test parsing multiple articles."""
    
    # Look for sample HTML files in the data directory
    data_dir = Path(__file__).parent / "src" / "cdc_text_corpora" / "data"
    
    # Try to find sample PCD files
    sample_files = []
    pcd_dir = data_dir / "pcd" / "issues"
    if pcd_dir.exists():
        for year_dir in pcd_dir.iterdir():
            if year_dir.is_dir():
                for html_file in year_dir.glob("*.htm"):
                    # Skip non-English versions
                    if not (html_file.name.endswith("_es.htm") or 
                           html_file.name.endswith("_fr.htm") or
                           html_file.name.endswith("_zhs.htm") or
                           html_file.name.endswith("_zht.htm")):
                        sample_files.append(html_file)
    
    if not sample_files:
        print("No sample HTML files found in data directory")
        return
    
    # Set random seed for reproducible results
    random.seed(43)
    
    # Randomly select 10 files
    selected_files = random.sample(sample_files, min(10, len(sample_files)))
    
    print(f"Found {len(sample_files)} total files, testing with {len(selected_files)} randomly selected files")
    
    # Initialize parser
    parser = CDCArticleParser('pcd', 'journal', 'en')
    
    # Test each article
    for i, sample_file in enumerate(selected_files):
        print(f"\n{'='*60}")
        print(f"TESTING ARTICLE {i+1}: {sample_file.name}")
        print('='*60)
        
        try:
            html_content = parser._load_html_file(str(sample_file))
            article = parser.parse_article(str(sample_file), html_content)
            
            # Display results
            print(f"Title: {article.title}")
            print(f"Authors: {article.authors}")
            print(f"Publication Date: {article.publication_date}")
            print(f"Abstract: {article.abstract[:150]}..." if article.abstract else "Abstract: None")
            print(f"Full Text Length: {len(article.full_text)} characters")
            print(f"References: {len(article.references)} found")
            
            print("✓ PARSING SUCCESSFUL!")
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

if __name__ == "__main__":
    test_single_article_parsing()