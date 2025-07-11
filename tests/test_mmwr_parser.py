#!/usr/bin/env python3
"""Unit tests for MMWR article parser."""

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


def get_mmwr_url_path(filename):
    """Convert MMWR filename to expected URL path."""
    if filename.startswith("mmwr_volumes_"):
        # e.g., mmwr_volumes_71_wr_mm7153a1.htm -> mmwr/volumes/71/wr/mm7153a1.htm
        parts = filename.replace("mmwr_volumes_", "").replace(".htm", "").split("_")
        if len(parts) >= 3:
            volume = parts[0]
            series = parts[1]  # wr, ss, etc.
            article = "_".join(parts[2:])  # mm7153a1, ss7208a1, etc.
            return f"mmwr/volumes/{volume}/{series}/{article}.htm"
    elif filename.startswith("mmwr_preview_mmwrhtml_"):
        # e.g., mmwr_preview_mmwrhtml_mm6452a3.htm -> mmwr/preview/mmwrhtml/mm6452a3.htm
        article = filename.replace("mmwr_preview_mmwrhtml_", "").replace(".htm", "")
        return f"mmwr/preview/mmwrhtml/{article}.htm"
    
    # Fallback - shouldn't happen with proper filenames
    return filename.replace("mmwr_", "mmwr/").replace("_", "/")


class TestMMWRParser:
    """Test cases for the MMWR article parser."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent / "articles"
    
    @pytest.fixture
    def mmwr_expected_outputs(self, test_data_dir):
        """Get the expected outputs from the MMWR JSON file."""
        json_file = test_data_dir / "mmwr" / "output" / "mmwr.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert to dict keyed by filename for easy lookup
            return {article["file"]: article for article in data["articles"]}
    
    @pytest.fixture
    def parser(self):
        """Create a MMWR parser instance for testing."""
        return create_parser(
            collection='mmwr',
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
        "mmwr_volumes_71_wr_mm7153a1.htm",
        "mmwr_volumes_68_wr_mm6844e1.htm",
        "mmwr_volumes_71_wr_mm7150a5.htm",
        "mmwr_volumes_71_wr_mm7150a8.htm",
        "mmwr_volumes_71_wr_mm715152a2.htm",
        "mmwr_volumes_72_ss_ss7208a1.htm",
        "mmwr_preview_mmwrhtml_rr5416a1.htm",
        "mmwr_preview_mmwrhtml_mm6452a3.htm",
        "mmwr_preview_mmwrhtml_mm5650a4.htm",
        "mmwr_preview_mmwrhtml_mm5650a1.htm"
    ])
    def test_mmwr_parser_basic_fields(self, parser, test_data_dir, mmwr_expected_outputs, filename):
        """Test that MMWR parser correctly extracts basic fields (url, title, abstract)."""
        expected_url_path = get_mmwr_url_path(filename)
        
        # Load the HTML file
        html_file = test_data_dir / "mmwr" / "html" / filename
        html_content = self._load_html_file(html_file)
        
        # Parse the article
        article = parser.parse_article(expected_url_path, html_content)
        
        # Get expected output for this file
        expected = mmwr_expected_outputs[filename]
        
        # Assert URL
        assert article.url == expected['url'], f"Expected URL: {expected['url']}, got: {article.url}"
        
        # Assert title (using similarity matching for robustness with unicode differences)
        expected_title = expected['title']
        if expected_title == "":
            # If expected is empty, got should be empty too
            assert article.title == "", f"Expected empty title, got: {article.title}"
        else:
            # Use similarity matching with 0.95 ratio for non-empty titles
            similarity = similarity_ratio(article.title, expected_title)
            assert similarity >= 0.95, f"Title similarity {similarity:.3f} < 0.95. Expected: {expected_title}, got: {article.title}"
        
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
        "mmwr_volumes_71_wr_mm7153a1.htm",
        "mmwr_volumes_68_wr_mm6844e1.htm",
        "mmwr_volumes_71_wr_mm7150a5.htm",
        "mmwr_volumes_71_wr_mm7150a8.htm",
        "mmwr_volumes_71_wr_mm715152a2.htm",
        "mmwr_volumes_72_ss_ss7208a1.htm",
        "mmwr_preview_mmwrhtml_rr5416a1.htm",
        "mmwr_preview_mmwrhtml_mm6452a3.htm",
        "mmwr_preview_mmwrhtml_mm5650a4.htm",
        "mmwr_preview_mmwrhtml_mm5650a1.htm"
    ])
    def test_mmwr_parser_metadata(self, parser, test_data_dir, filename):
        """Test that MMWR parser correctly sets metadata fields."""
        expected_url_path = get_mmwr_url_path(filename)
        
        html_file = test_data_dir / "mmwr" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        # Assert metadata fields
        assert article.relative_url == expected_url_path
        assert article.journal == 'mmwr'
        assert article.language == 'en'
        assert article.collection == CDCCollections.MMWR
        assert article.html_text == html_content
    
    @pytest.mark.parametrize("filename", [
        "mmwr_volumes_71_wr_mm7153a1.htm",
        "mmwr_volumes_68_wr_mm6844e1.htm",
        "mmwr_volumes_71_wr_mm7150a5.htm",
        "mmwr_volumes_71_wr_mm7150a8.htm",
        "mmwr_volumes_71_wr_mm715152a2.htm",
        "mmwr_volumes_72_ss_ss7208a1.htm",
        "mmwr_preview_mmwrhtml_rr5416a1.htm",
        "mmwr_preview_mmwrhtml_mm6452a3.htm",
        "mmwr_preview_mmwrhtml_mm5650a4.htm",
        "mmwr_preview_mmwrhtml_mm5650a1.htm"
    ])
    def test_mmwr_parser_content_not_empty(self, parser, test_data_dir, filename):
        """Test that MMWR parser extracts non-empty content."""
        expected_url_path = get_mmwr_url_path(filename)
        
        html_file = test_data_dir / "mmwr" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        # Assert that content fields are not empty
        assert article.title.strip() != "", "Title should not be empty"
        # Note: Abstract can be empty for some article types
        assert article.full_text.strip() != "", "Full text should not be empty"
        assert article.url.strip() != "", "URL should not be empty"
        
        # Assert that HTML content is preserved
        assert len(article.html_text) > 0, "HTML text should not be empty"
    
    @pytest.mark.parametrize("filename", [
        "mmwr_volumes_71_wr_mm7153a1.htm",
        "mmwr_volumes_68_wr_mm6844e1.htm",
        "mmwr_volumes_71_wr_mm7150a5.htm",
        "mmwr_volumes_71_wr_mm7150a8.htm",
        "mmwr_volumes_71_wr_mm715152a2.htm",
        "mmwr_volumes_72_ss_ss7208a1.htm",
        "mmwr_preview_mmwrhtml_rr5416a1.htm",
        "mmwr_preview_mmwrhtml_mm6452a3.htm",
        "mmwr_preview_mmwrhtml_mm5650a4.htm",
        "mmwr_preview_mmwrhtml_mm5650a1.htm"
    ])
    def test_mmwr_parser_return_type(self, parser, test_data_dir, filename):
        """Test that MMWR parser returns an Article object."""
        expected_url_path = get_mmwr_url_path(filename)
        
        html_file = test_data_dir / "mmwr" / "html" / filename
        html_content = self._load_html_file(html_file)
        article = parser.parse_article(expected_url_path, html_content)
        
        assert isinstance(article, Article), "MMWR parser should return an Article object"
    
    def test_mmwr_parser_with_empty_html(self, parser):
        """Test that MMWR parser handles empty HTML gracefully."""
        relative_url = "mmwr/volumes/71/wr/empty.htm"
        article = parser.parse_article(relative_url, "")
        
        assert isinstance(article, Article), "MMWR parser should return an Article object even with empty HTML"
        assert article.relative_url == relative_url
        assert article.journal == 'mmwr'
        assert article.language == 'en'
        assert article.collection == CDCCollections.MMWR


def test_mmwr_parser_standalone():
    """Standalone test function for MMWR parser that can be run directly."""
    test_data_dir = Path(__file__).parent / "articles"
    mmwr_expected_file = test_data_dir / "mmwr" / "output" / "mmwr.json"
    
    if not mmwr_expected_file.exists():
        print(f"Expected output file not found: {mmwr_expected_file}")
        return False
    
    # Load expected outputs
    with open(mmwr_expected_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        expected_outputs = {article["file"]: article for article in data["articles"]}
    
    # Test all files from mmwr.json
    test_files = [
        "mmwr_volumes_71_wr_mm7153a1.htm",
        "mmwr_volumes_68_wr_mm6844e1.htm",
        "mmwr_volumes_71_wr_mm7150a5.htm",
        "mmwr_volumes_71_wr_mm7150a8.htm",
        "mmwr_volumes_71_wr_mm715152a2.htm",
        "mmwr_volumes_72_ss_ss7208a1.htm",
        "mmwr_preview_mmwrhtml_rr5416a1.htm",
        "mmwr_preview_mmwrhtml_mm6452a3.htm",
        "mmwr_preview_mmwrhtml_mm5650a4.htm",
        "mmwr_preview_mmwrhtml_mm5650a1.htm"
    ]
    
    all_success = True
    
    for filename in test_files:
        relative_url = get_mmwr_url_path(filename)
    
        print(f"{'='*60}")
        print(f"TESTING MMWR: {filename}")
        print('='*60)
        
        # Check if test file exists
        test_file = test_data_dir / "mmwr" / "html" / filename
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
            collection='mmwr',
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
            
            # Check title (using similarity matching)
            expected_title = expected['title']
            if expected_title == "":
                # If expected is empty, got should be empty too
                if article.title == "":
                    print(f"✅ Title matches (empty): {article.title}")
                else:
                    print(f"❌ Expected empty title, got: {article.title}")
                    file_success = False
            else:
                # Use similarity matching with 0.95 ratio for non-empty titles
                similarity = similarity_ratio(article.title, expected_title)
                if similarity >= 0.95:
                    print(f"✅ Title matches (similarity: {similarity:.3f}): {article.title}")
                else:
                    print(f"❌ Title similarity {similarity:.3f} < 0.95:")
                    print(f"  Expected: '{expected_title}'")
                    print(f"  Got: '{article.title}'")
                    file_success = False
            
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
    success = test_mmwr_parser_standalone()
    if success:
        print("\n🎉 All MMWR tests passed!")
    else:
        print("\n❌ Some MMWR tests failed!")
    sys.exit(0 if success else 1)