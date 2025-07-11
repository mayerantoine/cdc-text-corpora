#!/usr/bin/env python3
"""Unit tests for EID article parser."""

import sys
import os
import json
import pytest
from pathlib import Path
from difflib import SequenceMatcher

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cdc_text_corpora.core.parser import create_parser, Article, CDCCollections


def similarity_ratio(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


class TestEIDParser:
    """Test cases for the EID article parser."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent / "articles"
    
    @pytest.fixture
    def eid_expected_outputs(self, test_data_dir):
        """Get the expected outputs from the EID JSON file."""
        json_file = test_data_dir / "eid" / "output" / "eid.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert to dict keyed by filename for easy lookup
            return {article["file"]: article for article in data["articles"]}
    
    @pytest.fixture
    def parser(self):
        """Create an EID parser instance for testing."""
        return create_parser(
            collection='eid',
            journal_type='journal',
            language='en',
            articles_collection={},
            validate_articles=False
        )
    
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
    
    @pytest.mark.parametrize("filename", [
        "eid_article_11_10_05-0508_article.htm",
        "eid_article_18_5_11-1111_article.htm",
        "eid_article_18_5_11-1275_article.htm",
        "eid_article_21_8_14-0956_article.htm",
        "eid_article_21_8_14-1251_article.htm",
        "eid_article_21_8_15-0423_article.htm",
        "eid_article_21_8_et-2108_article.htm"
    ])
    def test_eid_parser_basic_fields(self, parser, test_data_dir, eid_expected_outputs, filename):
        """Test that EID parser correctly extracts basic fields (url, title, abstract)."""
        # Convert filename to expected URL path (e.g., eid_article_11_10_05-0508_article.htm -> eid/article/11/10/05-0508_article.htm)
        parts = filename.replace("eid_article_", "").replace(".htm", "").split("_")
        expected_url_path = f"eid/article/{parts[0]}/{parts[1]}/{parts[2]}.htm"
        
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
        
        # Assert abstract (using similarity matching for robustness)
        expected_abstract = expected['abstract']
        if expected_abstract == "":
            # If expected is empty, got should be empty too
            assert article.abstract == "", f"Expected empty abstract, got: {article.abstract}"
        else:
            # Use similarity matching with 0.95 ratio for non-empty abstracts
            similarity = similarity_ratio(article.abstract, expected_abstract)
            assert similarity >= 0.95, f"Abstract similarity {similarity:.3f} < 0.95. Expected: {expected_abstract[:100]}..., got: {article.abstract[:100]}..."
    
    @pytest.mark.parametrize("filename", [
        "eid_article_11_10_05-0508_article.htm",
        "eid_article_18_5_11-1111_article.htm",
        "eid_article_18_5_11-1275_article.htm",
        "eid_article_21_8_14-0956_article.htm",
        "eid_article_21_8_14-1251_article.htm",
        "eid_article_21_8_15-0423_article.htm",
        "eid_article_21_8_et-2108_article.htm"
    ])
    def test_eid_parser_metadata(self, parser, test_data_dir, filename):
        """Test that EID parser correctly sets metadata fields."""
        # Convert filename to expected URL path
        parts = filename.replace("eid_article_", "").replace(".htm", "").split("_")
        expected_url_path = f"eid/article/{parts[0]}/{parts[1]}/{parts[2]}.htm"
        
        html_file = test_data_dir / "eid" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        # Assert metadata fields
        assert article.relative_url == expected_url_path
        assert article.journal == 'eid'
        assert article.language == 'en'
        assert article.collection == CDCCollections.EID
        assert article.html_text == html_content
    
    @pytest.mark.parametrize("filename", [
        "eid_article_11_10_05-0508_article.htm",
        "eid_article_18_5_11-1111_article.htm",
        "eid_article_18_5_11-1275_article.htm",
        "eid_article_21_8_14-0956_article.htm",
        "eid_article_21_8_14-1251_article.htm",
        "eid_article_21_8_15-0423_article.htm",
        "eid_article_21_8_et-2108_article.htm"
    ])
    def test_eid_parser_content_not_empty(self, parser, test_data_dir, filename):
        """Test that EID parser extracts non-empty content."""
        # Convert filename to expected URL path
        parts = filename.replace("eid_article_", "").replace(".htm", "").split("_")
        expected_url_path = f"eid/article/{parts[0]}/{parts[1]}/{parts[2]}.htm"
        
        html_file = test_data_dir / "eid" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        # Assert that content fields are not empty
        assert article.title.strip() != "", "Title should not be empty"
        # Note: Abstract can be empty for some article types (like Etymologia articles)
        assert article.full_text.strip() != "", "Full text should not be empty"
        assert article.url.strip() != "", "URL should not be empty"
        
        # Assert that HTML content is preserved
        assert len(article.html_text) > 0, "HTML text should not be empty"
    
    @pytest.mark.parametrize("filename", [
        "eid_article_11_10_05-0508_article.htm",
        "eid_article_18_5_11-1111_article.htm",
        "eid_article_18_5_11-1275_article.htm",
        "eid_article_21_8_14-0956_article.htm",
        "eid_article_21_8_14-1251_article.htm",
        "eid_article_21_8_15-0423_article.htm",
        "eid_article_21_8_et-2108_article.htm"
    ])
    def test_eid_parser_return_type(self, parser, test_data_dir, filename):
        """Test that EID parser returns an Article object."""
        # Convert filename to expected URL path
        parts = filename.replace("eid_article_", "").replace(".htm", "").split("_")
        expected_url_path = f"eid/article/{parts[0]}/{parts[1]}/{parts[2]}.htm"
        
        html_file = test_data_dir / "eid" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        assert isinstance(article, Article), "EID parser should return an Article object"
    
    def test_eid_parser_with_empty_html(self, parser):
        """Test that EID parser handles empty HTML gracefully."""
        relative_url = "eid/article/empty.htm"
        article = parser.parse_article(relative_url, "")
        
        assert isinstance(article, Article), "EID parser should return an Article object even with empty HTML"
        assert article.relative_url == relative_url
        assert article.journal == 'eid'
        assert article.language == 'en'
        assert article.collection == CDCCollections.EID


def test_eid_parser_standalone():
    """Standalone test function for EID parser that can be run directly."""
    test_data_dir = Path(__file__).parent / "articles"
    eid_expected_file = test_data_dir / "eid" / "output" / "eid.json"
    
    if not eid_expected_file.exists():
        print(f"Expected output file not found: {eid_expected_file}")
        return False
    
    # Load expected outputs
    with open(eid_expected_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        expected_outputs = {article["file"]: article for article in data["articles"]}
    
    # Test files from eid.json
    test_files = [
        "eid_article_11_10_05-0508_article.htm",
        "eid_article_18_5_11-1111_article.htm",
        "eid_article_18_5_11-1275_article.htm",
        "eid_article_21_8_14-0956_article.htm",
        "eid_article_21_8_14-1251_article.htm",
        "eid_article_21_8_15-0423_article.htm",
        "eid_article_21_8_et-2108_article.htm"
    ]
    
    all_success = True
    
    for filename in test_files:
        # Convert filename to expected URL path
        parts = filename.replace("eid_article_", "").replace(".htm", "").split("_")
        relative_url = f"eid/article/{parts[0]}/{parts[1]}/{parts[2]}.htm"
        
        print(f"{'='*60}")
        print(f"TESTING EID: {filename}")
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
        
        # Create parser and parse article
        parser = create_parser(
            collection='eid',
            journal_type='journal',
            language='en',
            articles_collection={},
            validate_articles=False
        )
        
        try:
            article = parser.parse_article(relative_url, html_content)
            expected = expected_outputs[filename]
            
            file_success = True
            
            # Check URL
            if article.url != expected['url']:
                print(f"❌ URL mismatch: Expected '{expected['url']}', got '{article.url}'")
                file_success = False
            else:
                print(f"✅ URL matches: {article.url}")
            
            # Check title
            if article.title != expected['title']:
                print(f"❌ Title mismatch: Expected '{expected['title']}', got '{article.title}'")
                file_success = False
            else:
                print(f"✅ Title matches: {article.title}")
            
            # Check abstract (using similarity matching)
            expected_abstract = expected['abstract']
            if expected_abstract == "":
                # If expected is empty, got should be empty too
                if article.abstract == "":
                    print(f"✅ Abstract matches (empty): {article.abstract}")
                else:
                    print(f"❌ Expected empty abstract, got: {article.abstract[:100]}...")
                    file_success = False
            else:
                # Use similarity matching with 0.95 ratio for non-empty abstracts
                similarity = similarity_ratio(article.abstract, expected_abstract)
                if similarity >= 0.95:
                    print(f"✅ Abstract matches (similarity: {similarity:.3f}): {article.abstract[:100]}...")
                else:
                    print(f"❌ Abstract similarity {similarity:.3f} < 0.95:")
                    print(f"  Expected length: {len(expected_abstract)}")
                    print(f"  Got length: {len(article.abstract)}")
                    print(f"  Expected: '{expected_abstract[:200]}...'")
                    print(f"  Got: '{article.abstract[:200]}...'")
                    file_success = False
            
            if not file_success:
                all_success = False
            
        except Exception as e:
            print(f"❌ Error parsing article: {e}")
            all_success = False
    
    return all_success


if __name__ == "__main__":
    success = test_eid_parser_standalone()
    if success:
        print("\n🎉 All EID tests passed!")
    else:
        print("\n❌ Some EID tests failed!")
    sys.exit(0 if success else 1)