#!/usr/bin/env python3
"""Unit tests for CDC article parser parse_article function."""

import sys
import os
import json
import pytest
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cdc_text_corpora.core.parser import CDCArticleParser, Article, CDCCollections


class TestParseArticle:
    """Test cases for the parse_article function."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent / "articles"
    
    @pytest.fixture
    def eid_test_file(self, test_data_dir):
        """Get the EID test HTML file."""
        return test_data_dir / "eid" / "html" / "eid_article_11_10_05-0508_article.htm"
    
    @pytest.fixture
    def eid_expected_outputs(self, test_data_dir):
        """Get the expected outputs from the JSON file."""
        json_file = test_data_dir / "eid" / "output" / "eid.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert to dict keyed by filename for easy lookup
            return {article["file"]: article for article in data["articles"]}
    
    @pytest.fixture
    def eid_html_content(self, eid_test_file):
        """Load the HTML content from the test file."""
        encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(eid_test_file, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError("Unable to decode test file with any encoding")
    
    @pytest.fixture
    def parser(self):
        """Create a CDCArticleParser instance for testing."""
        return CDCArticleParser(
            journal='eid',
            journal_type='journal',
            language='en',
            articles_collection={},
            validate_articles=False
        )
    
    @pytest.mark.parametrize("filename,expected_url_path", [
        ("eid_article_11_10_05-0508_article.htm", "eid/article/11/10/05-0508_article.htm"),
        ("eid_article_18_5_11-1111_article.htm", "eid/article/18/5/11-1111_article.htm"),
        ("eid_article_18_5_11-1275_article.htm", "eid/article/18/5/11-1275_article.htm"),
    ])
    def test_parse_article_basic_fields(self, parser, test_data_dir, eid_expected_outputs, filename, expected_url_path):
        """Test that parse_article correctly extracts basic fields (url, title, abstract)."""
        # Load the HTML file
        html_file = test_data_dir / "eid" / "html" / filename
        html_content = self._load_html_file(html_file)
        
        # Parse the article
        article = parser.parse_article(expected_url_path, html_content)
        
        # Get expected output for this file
        expected = eid_expected_outputs[filename]
        
        # Assert URL
        assert article.url == expected['url'], f"Expected URL: {expected['url']}, got: {article.url}"
        
        # Assert title
        assert article.title == expected['title'], f"Expected title: {expected['title']}, got: {article.title}"
        
        # Assert abstract
        assert article.abstract == expected['abstract'], f"Expected abstract: {expected['abstract']}, got: {article.abstract}"
    
    def _load_html_file(self, file_path):
        """Helper method to load HTML content from file."""
        encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError("Unable to decode test file with any encoding")
    
    def test_parse_article_metadata(self, parser, eid_html_content):
        """Test that parse_article correctly sets metadata fields."""
        relative_url = "eid/article/11/10/05-0508_article.htm"
        article = parser.parse_article(relative_url, eid_html_content)
        
        # Assert metadata fields
        assert article.relative_url == relative_url
        assert article.journal == 'eid'
        assert article.language == 'en'
        assert article.collection == CDCCollections.EID
        assert article.html_text == eid_html_content
    
    def test_parse_article_content_not_empty(self, parser, eid_html_content):
        """Test that parse_article extracts non-empty content."""
        relative_url = "eid/article/11/10/05-0508_article.htm"
        article = parser.parse_article(relative_url, eid_html_content)
        
        # Assert that content fields are not empty
        assert article.title.strip() != "", "Title should not be empty"
        assert article.abstract.strip() != "", "Abstract should not be empty"
        assert article.full_text.strip() != "", "Full text should not be empty"
        assert article.url.strip() != "", "URL should not be empty"
        
        # Assert that HTML content is preserved
        assert len(article.html_text) > 0, "HTML text should not be empty"
    
    def test_parse_article_return_type(self, parser, eid_html_content):
        """Test that parse_article returns an Article object."""
        relative_url = "eid/article/11/10/05-0508_article.htm"
        article = parser.parse_article(relative_url, eid_html_content)
        
        assert isinstance(article, Article), "parse_article should return an Article object"
    
    def test_parse_article_with_empty_html(self, parser):
        """Test that parse_article handles empty HTML gracefully."""
        relative_url = "eid/article/empty.htm"
        article = parser.parse_article(relative_url, "")
        
        assert isinstance(article, Article), "parse_article should return an Article object even with empty HTML"
        assert article.relative_url == relative_url
        assert article.journal == 'eid'
        assert article.language == 'en'
        assert article.collection == CDCCollections.EID


def test_parse_article_standalone():
    """Standalone test function that can be run directly."""
    test_data_dir = Path(__file__).parent / "articles"
    eid_expected_file = test_data_dir / "eid" / "output" / "eid.json"
    
    if not eid_expected_file.exists():
        print(f"Expected output file not found: {eid_expected_file}")
        return False
    
    # Load expected outputs
    with open(eid_expected_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        expected_outputs = {article["file"]: article for article in data["articles"]}
    
    # Define test cases
    test_cases = [
        ("eid_article_11_10_05-0508_article.htm", "eid/article/11/10/05-0508_article.htm"),
        ("eid_article_18_5_11-1111_article.htm", "eid/article/18/5/11-1111_article.htm"),
        ("eid_article_18_5_11-1275_article.htm", "eid/article/18/5/11-1275_article.htm"),
    ]
    
    # Create parser
    parser = CDCArticleParser(
        journal='eid',
        journal_type='journal',
        language='en',
        articles_collection={},
        validate_articles=False
    )
    
    all_success = True
    
    for filename, relative_url in test_cases:
        print(f"\n{'='*60}")
        print(f"TESTING: {filename}")
        print('='*60)
        
        # Check if test file exists
        test_file = test_data_dir / "eid" / "html" / filename
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            all_success = False
            continue
        
        # Load HTML content
        html_content = None
        encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(test_file, encoding=encoding) as f:
                    html_content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if html_content is None:
            print(f"❌ Unable to decode test file with any encoding")
            all_success = False
            continue
        
        # Parse article
        try:
            article = parser.parse_article(relative_url, html_content)
            expected = expected_outputs[filename]
            
            # Check URL
            if article.url != expected['url']:
                print(f"❌ URL mismatch: Expected '{expected['url']}', got '{article.url}'")
                all_success = False
            else:
                print(f"✅ URL matches: {article.url}")
            
            # Check title
            if article.title != expected['title']:
                print(f"❌ Title mismatch: Expected '{expected['title']}', got '{article.title}'")
                all_success = False
            else:
                print(f"✅ Title matches: {article.title}")
            
            # Check abstract
            if article.abstract != expected['abstract']:
                print(f"❌ Abstract mismatch: Expected '{expected['abstract'][:100]}...', got '{article.abstract[:100]}...'")
                all_success = False
            else:
                print(f"✅ Abstract matches: {article.abstract[:100]}...")
                
        except Exception as e:
            print(f"❌ Error parsing article: {e}")
            all_success = False
    
    return all_success


if __name__ == "__main__":
    success = test_parse_article_standalone()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    sys.exit(0 if success else 1)