from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
from parsel import Selector
import os
import pathlib
import json
import re
from zipfile import ZipFile
from tqdm.auto import tqdm
from enum import Enum
from abc import ABC, abstractmethod
from cdc_text_corpora.utils.config import get_data_directory, get_collection_zip_path


class CDCCollections(Enum):
    """Enum for the three fixed CDC collections."""
    PCD = "pcd"  # Preventing Chronic Disease
    EID = "eid"  # Emerging Infectious Diseases
    MMWR = "mmwr"  # Morbidity and Mortality Weekly Report


@dataclass
class Article:
    """Class to represent a structured CDC article."""
    
    title: str = ""
    abstract: str = ""
    full_text: str = ""  # Added field for full article content when no standard sections exist
    references: List[str] = field(default_factory=list)
    html_text: str = ""
    url: str = ""
    relative_url: str = "" # current relative path of the htm file
    journal: str = ""
    language: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: str = ""
    collection: Optional[CDCCollections] = None
 

class HTMLArticleLoader:
    def __init__(self, journal: str, journal_type: str, language: str):
        self.journal = journal
        self.journal_type = journal_type
        self.language = language
        self.articles_html: Optional[Dict[str, str]] = None

    def load_from_file(self) -> None:
        self.articles_html  = self._load_html_collection()

    def _load_html_collection(self) -> Dict[str, str]:
        """Load HTML content for the specified journal by scanning HTML files in collection-specific folders.
        
        Collection folder structures:
        - PCD: uses 'issues' folder
        - EID: uses 'articles' folder  
        - MMWR: uses 'preview/mmwrhtml' and 'volumes' folders
        
        Filters by language:
        - 'en': Load only English files (no language suffix)
        - 'es': Load only Spanish files (_es.htm suffix)
        - 'fr': Load only French files (_fr.htm suffix)
        - 'zhs': Load only Simplified Chinese files (_zhs.htm suffix)
        - 'zht': Load only Traditional Chinese files (_zht.htm suffix)
        - None or empty: Load all languages
        """
        data_dir = get_data_directory()
        collection_base_dir = data_dir / "json-html" / self.journal
        
        # Define collection-specific folders
        collection_folders = {
            'pcd': ['issues'],
            'eid': ['article'], 
            'mmwr': ['preview/mmwrhtml', 'volumes']
        }
        
        if self.journal.lower() not in collection_folders:
            print(f"Unknown collection: {self.journal}")
            return {}
        
        folders_to_scan = collection_folders[self.journal.lower()]
        articles_html = {}
        
        # Check if collection base directory exists
        if not collection_base_dir.exists():
            print(f"Collection directory not found: {collection_base_dir}")
            # Try to extract from zip if directory doesn't exist
            self._extract_zip_files()
            if not collection_base_dir.exists():
                print(f"Unable to find HTML content for {self.journal}")
                return {}
        
        # Process each folder for this collection
        for folder_path in folders_to_scan:
            target_dir = collection_base_dir / folder_path
            
            if not target_dir.exists():
                print(f"Folder not found: {target_dir}, skipping...")
                continue
            
            print(f"Scanning folder: {target_dir}")
            
            # Walk through all directories and subdirectories in the target folder
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.endswith('.htm') or file.endswith('.html'):
                        file_path = pathlib.Path(root) / file
                        relative_path = file_path.relative_to(collection_base_dir)
                        
                        # Filter out non-content files
                        file_lower = file.lower()
                        if ('cover' in file_lower or 
                            'ac-' in file_lower or
                            'toc' in file_lower or
                            'index' in file_lower or
                            'archive' in file_lower):
                            continue
                        
                        # filter Erratum
                        if file.endswith('e.htm'):
                            continue

                        # Filter based on language if specified
                        if self.language == 'en':
                            # Skip non-English files (those ending with language codes)
                            if file.endswith(('_es.htm', '_fr.htm', '_zhs.htm', '_zht.htm')):
                                continue
                        elif self.language == 'es':
                            # Only include Spanish files
                            if not file.endswith('_es.htm'):
                                continue
                        elif self.language == 'fr':
                            # Only include French files
                            if not file.endswith('_fr.htm'):
                                continue
                        elif self.language == 'zhs':
                            # Only include Simplified Chinese files
                            if not file.endswith('_zhs.htm'):
                                continue
                        elif self.language == 'zht':
                            # Only include Traditional Chinese files
                            if not file.endswith('_zht.htm'):
                                continue
                        # If no language specified or unrecognized language, load all files
                        # TODO get language for the file being loaded.
                        
                        try:
                            # Use the existing _load_html_file method for consistency
                            html_content = self._load_html_file(str(file_path))
                            # Use the relative path as the key to maintain structure
                            articles_html[str(relative_path)] = html_content
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
                            continue
        
        language_msg = f" ({self.language})" if self.language else " (all languages)"
        print(f"Loaded {len(articles_html)} HTML files for {self.journal}{language_msg}")
        return articles_html

    def _load_html_file(self, file_path_name: str) -> str:
        """Load HTML content for the specified paper.
        
        Tries multiple encodings to handle different file formats:
        1. UTF-8 (most common)
        2. unicode_escape (fallback for problematic files)
        3. latin-1 (final fallback)
        """
        encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path_name, encoding=encoding) as f:
                    html_content = f.read()
                return html_content
            except UnicodeDecodeError:
                # Try next encoding
                continue
            except FileNotFoundError:
                print(f"File not found: {file_path_name}")
                raise FileNotFoundError(f'Unable to find HTML file: {file_path_name}')
            except Exception as e:
                # For other errors, try next encoding
                print(f"Error reading {file_path_name} with {encoding}: {e}")
                continue
        
        # If all encodings failed
        raise UnicodeDecodeError(
            'multiple_encodings', 
            b'', 0, 0, 
            f'Unable to decode {file_path_name} with any of the tried encodings: {encodings_to_try}'
        )

    def _extract_zip_files(self) -> None:
        """Extract all collection zip files to the json-html directory."""
        data_dir = get_data_directory()
        zip_dir = data_dir / "html-outputs"
        html_dir = data_dir / "json-html"
        
        # Create the directory if it doesn't exist
        os.makedirs(html_dir, exist_ok=True)
        
        if not zip_dir.exists():
            print(f"Zip directory {zip_dir} does not exist")
            return
        
        for zip_file in os.listdir(zip_dir):
            if zip_file.endswith('.zip'):
                with ZipFile(zip_dir / zip_file, 'r') as zip_obj:
                    zip_obj.extractall(html_dir)


# TODO add a new module wtih a class that extends CDCArticleParser , but also create a VectorStore to index each parsed articles
# overwrite parsed all articles with its own method, no intermediate json parsed file
class CDCArticleParser(ABC):
    """Base class for CDC article parsers with common functionality."""

    def __init__(self, journal: str, journal_type: str, language: str, articles_collection: dict, validate_articles: bool = True):
        self.journal = journal
        self.journal_type = journal_type
        self.language = language
        self.articles_html = articles_collection
        self.validate_articles = validate_articles
        
        # Initialize validator if validation is enabled
        if self.validate_articles:
            from cdc_text_corpora.utils.validation import ArticleValidator
            self.validator: Optional['ArticleValidator'] = ArticleValidator()
        else:
            self.validator = None 
    
    @abstractmethod
    def parse_title(self, html: str) -> str:
        """Parse the article title from HTML. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def parse_authors(self, html: str) -> List[str]:
        """Parse author information from HTML. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def parse_abstract(self, html: str) -> str:
        """Parse the article abstract from HTML. Must be implemented by subclasses."""
        pass
    
    # Common utility methods
    def is_likely_institution(self, text: str) -> bool:
        """Check if text is likely an institution or department name."""
        institution_words = {
            'university', 'institute', 'center', 'division', 'department',
            'school', 'college', 'hospital', 'clinic', 'laboratory',
            'program', 'service', 'branch', 'office', 'bureau', 'center',
            'dept', 'div', 'services', 'unit', 'univ'
        }
        
        text_words = set(word.lower() for word in text.split())
        return bool(text_words & institution_words)

    def clean_author_name(self, text: str) -> str:
        """Clean author name by removing titles, degrees and affiliations."""
        # Skip if text looks like an institution
        if self.is_likely_institution(text):
            return ""
        
        # Remove common titles and degrees
        titles = ['Dr', 'PhD', 'MD', 'MPH', 'MS', 'MA', 'DrPH', 'MSc', 'MBBS', 'DVM', 'BSc', 'MPP', 'ScD']
        text = text.strip()
        
        # Remove anything in parentheses first
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Remove titles with word boundaries
        for title in titles:
            text = re.sub(rf'\b{title}\b\.?', '', text)
        
        # Split on institutional indicators and take first part
        splits = re.split(r'\b(?:from|at|of)\b', text, flags=re.IGNORECASE)
        text = splits[0]
        
        # Remove anything after a comma or semicolon
        text = text.split(',')[0].split(';')[0]
        
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = text.strip(' .,;')
        
        # Final check - if it looks like an institution after cleaning, return empty
        if self.is_likely_institution(text):
            return ""
            
        return text

    def clean_text_with_notices(self, text: str) -> str:
        """Clean text by removing common notice/navigation text."""
        notices = [
            "Persons using assistive technology might not be able to fully access information in this file",
            "For assistance, please send e-mail to",
            "Type 508 Accommodation",
            "mmwrq@cdc.gov",
            "Information about electronic access to this publication",
            "Back to top",
            "Page last reviewed:",
            "Page last updated:",
            "Content source:",
            "508 Accommodation and the title",
            "Top of Page"
        ]
        
        for notice in notices:
            text = text.replace(notice, '')
        
        # Remove navigation text
        text = re.sub(r'Previous\s+Page|Next\s+Page|Table\s+of\s+Contents|Top\s+of\s+Page', '', text, flags=re.IGNORECASE)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def get_authors_from_meta(self, selector: Selector) -> List[str]:
        """Get authors from meta tags (common in PCD)."""
        authors = []
        meta_authors = selector.xpath('//meta[@name="citation_author"]/@content').getall()
        if meta_authors:
            for author in meta_authors:
                clean_name = author.strip()
                if clean_name:
                    authors.append(clean_name)
        return authors


    def filter_articles(self) -> Dict[str, str]:
        """Filter out cover articles and other non-content pages."""
        filtered: Dict[str, str] = {}
        if not self.articles_html:
            return filtered
        
        for url, html in self.articles_html.items():
            # Skip other non-content files
            if ('cover' not in url.lower() and 
                'ac-' not in url.lower() and
                'toc' not in url.lower() and
                'index' not in url.lower() and
                'archive' not in url.lower()):
                filtered[url] = html
        
        return filtered

    def parse_full_text(self, html: str) -> str:
        """
        Extract all paragraph text from an article, regardless of sections.
        Used for articles without standard section structure.
        
        Handles journal-specific main content areas:
        - EID journals have content in a div with id="mainbody"
        - Filters out navigation, headers, and other non-content
        """
        selector = Selector(text=html)
        paragraphs: List[str] = []
        
        # For EID journals, first try extracting from mainbody div which contains the main article content
        if self.journal.startswith('eid') and selector.xpath('//div[@id="mainbody"]').get():
            for p in selector.xpath('//div[@id="mainbody"]//p'):
                # Get all text content within the paragraph, including nested elements
                text = " ".join(p.xpath('.//text()').getall())
                if text.strip():
                    paragraphs.append(text.strip())
        
        # If no paragraphs found (or not EID journal), extract from all paragraphs
        if not paragraphs:
            # Extract all text from paragraphs, including text within nested elements (like <em>, <strong>, etc.)
            for p in selector.xpath('//p'):
                # Get all text content within the paragraph, including nested elements
                text = " ".join(p.xpath('.//text()').getall())
                if text.strip():
                    paragraphs.append(text.strip())
        
        # Filter out navigation elements and other common non-content text
        navigation_phrases = [
            'Top', 'Back to top', 'Top of Page', 'Next Page', 'Previous Page',
            'Exit Notification', 'Disclaimer', 'Privacy Policy', 'Accessibility',
            'CDC Home', 'Search', 'Health Topics', 'Contact Us'
        ]
        
        clean_paragraphs = []
        for p in paragraphs:
            p_clean = p.replace("\n", " ").strip()
            if p_clean and not any(nav in p_clean for nav in navigation_phrases):
                # Additional filtering for likely non-content paragraphs
                if len(p_clean) > 10 and not p_clean.endswith("..."):  # Avoid truncated content
                    clean_paragraphs.append(p_clean)
        
        # For MMWR, additional filtering may be needed for header/footer content
        if self.journal == 'mmwr':
            # MMWR often has publication info in short paragraphs at the beginning
            # Skip paragraphs that are publication metadata (usually shorter)
            if len(clean_paragraphs) > 3:
                # Check if first paragraphs are short (likely metadata)
                if len(clean_paragraphs[0]) < 100 and len(clean_paragraphs[1]) < 100:
                    clean_paragraphs = clean_paragraphs[2:]
        
        if not clean_paragraphs and len(paragraphs) > 0:
            # Fallback if filtering removed all content - use all non-empty paragraphs
            return " ".join([p.replace("\n", " ") for p in paragraphs if p.strip()])
            
        return " ".join(clean_paragraphs)

    def parse_references(self, html: str) -> List[str]:
        """Parse references from the HTML content."""
        selector = Selector(text=html)
        references = []
        
        # Look for references section
        ref_patterns = [
            "//h2[contains(text(), 'References')]/following-sibling::ol/li",
            "//h3[contains(text(), 'References')]/following-sibling::ol/li",
            "//h2[contains(text(), 'References')]/following-sibling::ul/li",
            "//h3[contains(text(), 'References')]/following-sibling::ul/li"
        ]
        
        for pattern in ref_patterns:
            ref_nodes = selector.xpath(pattern)
            if ref_nodes:
                for ref in ref_nodes:
                    ref_text = " ".join(ref.xpath('.//text()').getall()).strip()
                    if ref_text:
                        references.append(ref_text)
                break
        
        return references

    def parse_publication_date(self, html: str) -> str:
        """Parse publication date from the HTML content."""
        selector = Selector(text=html)
        
        # Common date patterns in CDC articles
        date_patterns = [
            r"(\w+ \d{1,2}, \d{4})",  # January 1, 2023
            r"(\d{1,2}/\d{1,2}/\d{4})",  # 1/1/2023
            r"(\d{4}-\d{2}-\d{2})",  # 2023-01-01
        ]
        
        # Look for dates in meta tags first
        meta_date = selector.xpath('//meta[@name="DC.date"]/@content').get()
        if meta_date:
            return meta_date
        
        # Look for dates in common locations
        date_locations = [
            "//p[contains(text(), 'Published') or contains(text(), 'Date')]//text()",
            "//div[contains(@class, 'date')]//text()",
            "//span[contains(@class, 'date')]//text()"
        ]
        
        for location in date_locations:
            texts = selector.xpath(location).getall()
            for text in texts:
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1)
        
        return ""

    def parse_url(self, html: str, relative_url: str = "") -> str:
        """Parse the full URL from HTML metadata or construct it from relative URL.
        
        Args:
            html: HTML content to parse
            relative_url: Relative URL path to use as fallback
            
        Returns:
            Full URL string
        """
        selector = Selector(text=html)
        
        # Try to get URL from meta tags first
        url_patterns = [
            '//meta[@property="og:url"]/@content',
            '//meta[@name="twitter:url"]/@content',
            '//meta[@name="DC.identifier"]/@content',
            '//link[@rel="canonical"]/@href'
        ]
        
        for pattern in url_patterns:
            url = selector.xpath(pattern).get()
            if url:
                # Clean and validate URL
                url = url.strip()
                if url.startswith(('http://', 'https://')):
                    return url
                elif url.startswith('/'):
                    # Relative URL from root
                    return f"https://www.cdc.gov{url}"
        
        # Fallback: construct URL from relative_url
        if relative_url:
            # Remove leading slash if present
            clean_relative_url = relative_url.lstrip('/')
            return f"https://www.cdc.gov/{clean_relative_url}"
        
        return ""

    def parse_article(self, relative_url: str, html: str) -> Article:
        """Parse a single article from HTML content."""
        article = Article()
        
        # Ensure relative_url is not empty and is a valid path
        if not relative_url or not relative_url.strip():
            raise ValueError(f"relative_url cannot be empty for article parsing")
        
        # Validate that relative_url looks like a valid file path
        relative_url_clean = relative_url.strip()
        if not relative_url_clean.endswith(('.htm', '.html')):
            raise ValueError(f"relative_url must be an HTML file path, got: {relative_url_clean}")
        
        # Set basic metadata
        article.relative_url = relative_url.strip()
        article.url = self.parse_url(html, relative_url)
        article.journal = self.journal
        article.language = self.language
        article.collection = CDCCollections(self.journal.lower())
        article.html_text = html 
        
        # Extract content using existing methods
        article.title = self.parse_title(html)
        article.abstract = self.parse_abstract(html)
        article.authors = self.parse_authors(html)
        article.full_text = self.parse_full_text(html)
        article.references = self.parse_references(html)
        article.publication_date = self.parse_publication_date(html)
        
        return article
  
    def parse_all_articles(self) -> Tuple[Dict[str, Article], Dict[str, Any]]:
        """Parse all articles in the loaded HTML content with validation.
        
        Returns:
            Tuple of (articles_dict, parsing_stats)
        """
        articles: Dict[str, Article] = {}
        filtered_html = self.filter_articles()
        
        if not filtered_html:
            print(f"No articles found to parse for {self.journal}")
            return articles, {"total_files": 0, "successful_parses": 0, "failed_parses": 0, "validation_stats": {}}
        
        print(f"Found {len(filtered_html)} articles to parse for {self.journal}")
        
        # Parsing counters
        successful_parses = 0
        failed_parses = 0
        
        # Validation counters
        valid_articles = 0
        invalid_articles = 0
        validation_issues = {"errors": 0, "warnings": 0, "info": 0}
        field_issues = {}
        
        for relative_url, html in tqdm(filtered_html.items(), desc=f"Parsing {self.journal} articles"):
            try:
                # Skip articles with empty or invalid relative URLs
                if not relative_url or not relative_url.strip():
                    print(f"Skipping article with empty relative_url")
                    failed_parses += 1
                    continue
                
                article = self.parse_article(relative_url, html)
                articles[relative_url] = article
                successful_parses += 1
                
                # Validate article if validation is enabled
                if self.validate_articles and self.validator:
                    from cdc_text_corpora.utils.validation import ValidationSeverity
                    validation_result = self.validator.validate_article(article)
                    
                    if validation_result.is_valid:
                        valid_articles += 1
                    else:
                        invalid_articles += 1
                    
                    # Count issues by severity
                    for issue in validation_result.issues:
                        if issue.severity == ValidationSeverity.ERROR:
                            validation_issues["errors"] += 1
                        elif issue.severity == ValidationSeverity.WARNING:
                            validation_issues["warnings"] += 1
                        elif issue.severity == ValidationSeverity.INFO:
                            validation_issues["info"] += 1
                        
                        # Count field-specific issues
                        if issue.field not in field_issues:
                            field_issues[issue.field] = 0
                        field_issues[issue.field] += 1
                
            except Exception as e:
                print(f"Error parsing {relative_url}: {e}")
                failed_parses += 1
                continue
        
        # Create comprehensive stats
        parsing_stats: Dict[str, Any] = {
            "total_files": len(filtered_html),
            "successful_parses": successful_parses,
            "failed_parses": failed_parses,
            "parsing_success_rate": (successful_parses / len(filtered_html)) * 100 if len(filtered_html) > 0 else 0,
        }
        
        if self.validate_articles:
            validation_stats = {
                "validation_enabled": True,
                "valid_articles": valid_articles,
                "invalid_articles": invalid_articles,
                "validation_rate": (valid_articles / successful_parses) * 100 if successful_parses > 0 else 0,
                "total_validation_issues": sum(validation_issues.values()),
                "issues_by_severity": validation_issues,
                "issues_by_field": field_issues
            }
        else:
            validation_stats = {"validation_enabled": False}
        
        parsing_stats["validation_stats"] = validation_stats
        
        # Print summary
        print(f"Parsing complete: {successful_parses} successful, {failed_parses} failed")
        if self.validate_articles:
            print(f"Validation: {valid_articles} valid, {invalid_articles} invalid articles")
            if sum(validation_issues.values()) > 0:
                print(f"Issues found: {validation_issues['errors']} errors, {validation_issues['warnings']} warnings, {validation_issues['info']} info")
        
        return articles, parsing_stats
    
    def save_as_json(self, articles: Dict[str, Article], parsing_stats: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> str:
        """Save parsed articles as JSON files in the json-parsed folder.
        
        Args:
            articles: Dictionary of parsed articles to save
            output_dir: Optional custom output directory. If None, uses default cdc-corpus-data/json-parsed
            
        Returns:
            Path to the saved JSON file
        """
        import json
        from datetime import datetime
        
        if not articles:
            print("No articles to save")
            return ""
        
        # Set up output directory
        if output_dir is None:
            data_dir = get_data_directory()
            output_base_dir = data_dir / "json-parsed"
        else:
            output_base_dir = pathlib.Path(output_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        language_suffix = f"_{self.language}" if self.language else "_all"
        filename = f"{self.journal}{language_suffix}_{timestamp}.json"
        output_file = output_base_dir / filename
        
        # Convert articles to JSON-serializable format
        articles_data = []
        for relative_url, article in articles.items():
            article_dict = {
                "relative_url": article.relative_url or relative_url,
                "url": article.url,
                "title": article.title,
                "abstract": article.abstract,
                "full_text": article.full_text,
                "html_text": article.html_text,
                "references": article.references,
                "journal": article.journal,
                "language": article.language,
                "authors": article.authors,
                "publication_date": article.publication_date,
                "collection": article.collection.value if article.collection else None
            }
            articles_data.append(article_dict)
        
        # Create metadata
        metadata = {
            "collection": self.journal,
            "language": self.language,
            "total_articles": len(articles_data),
            "generated_at": datetime.now().isoformat(),
            "parsing_stats": parsing_stats or {},
            "articles": articles_data
        }
        
        # Save to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(articles_data)} articles to {output_file}")
            return str(output_file)
            
        except Exception as e:
            print(f"❌ Error saving articles to JSON: {e}")
            raise


class EIDArticleParser(CDCArticleParser):
    """Parser for EID (Emerging Infectious Diseases) articles."""
    
    def parse_title(self, html: str) -> str:
        """Parse the article title from EID HTML."""
        selector = Selector(text=html)
        
        # Try to find EID-specific title format with class
        title_nodes = selector.xpath('//h3[@class="header article-title"]//text()').getall()
        if title_nodes:
            # Join all text nodes and clean up, excluding numeric references
            title_parts = []
            for text in title_nodes:
                text = text.strip()
                if text and not text.isdigit():  # Skip numeric references
                    title_parts.append(text)
            if title_parts:
                return " ".join(title_parts).strip()
        
        # For EID articles, try h3 with multiple text nodes (join all text within h3)
        h3_texts = selector.xpath('//h3[position()=1]//text()').getall()
        if h3_texts:
            # Join all text nodes and clean up
            title = " ".join(text.strip() for text in h3_texts if text.strip())
            if title:
                return title.strip()
        
        # Fallback - try first heading of any type
        title_result = selector.xpath('(//*[self::h1 or self::h2 or self::h3])[1]/text()').get()
        if title_result:
            return title_result.strip()
        
        return ""
    
    def parse_authors(self, html: str) -> List[str]:
        """Parse author information from EID HTML."""
        selector = Selector(text=html)
        authors = []
        
        # Try to get authors from meta tags first
        authors = self.get_authors_from_meta(selector)
        if authors:
            return authors
        
        # Try h4 with author class
        author_tags = selector.xpath("//h4[contains(@class, 'author')]//text()").getall()
        if author_tags:
            raw_authors = []
            current_author = []
            for tag in author_tags:
                if tag.strip():
                    if ',' in tag or 'and' in tag.lower():
                        # Split on delimiters and add to raw_authors
                        parts = re.split(r'[,;]|\band\b', tag)
                        raw_authors.extend(parts)
                    else:
                        current_author.append(tag)
            
            if current_author:
                raw_authors.append(' '.join(current_author))
            
            # Clean each author name
            for author in raw_authors:
                clean_name = self.clean_author_name(author)
                if clean_name and len(clean_name.split()) >= 2:
                    authors.append(clean_name)
        
        return authors
    
    def parse_abstract(self, html: str) -> str:
        """Parse the article abstract from EID HTML."""
        selector = Selector(text=html)
        
        # Try h3 with Abstract (common in EID journals)
        abstract_nodes = selector.xpath("//h3[contains(., 'Abstract')]/following-sibling::p//text()").getall()
        
        # If not found, try h2 with Abstract 
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//h2[contains(., 'Abstract')]/following-sibling::p//text()").getall()
        
        # If still not found, try div or section with abstract class
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//div[contains(@class, 'abstract')]//p//text()").getall()
        
        if abstract_nodes:
            abstract = " ".join(abstract_nodes)
            return self.clean_text_with_notices(abstract)
        
        return ""


class MMWRArticleParser(CDCArticleParser):
    """Parser for MMWR (Morbidity and Mortality Weekly Report) articles."""
    
    def parse_title(self, html: str) -> str:
        """Parse the article title from MMWR HTML."""
        selector = Selector(text=html)
        
        # Try to find title in h1 tags using normalize-space (both lowercase and uppercase - common in MMWR)
        title = selector.xpath('normalize-space(//h1)').get()
        if not title:
            title = selector.xpath('normalize-space(//H1)').get()
        if title:
            return title.strip()
        
        # Try HTML title tag as fallback
        title_tag = selector.xpath('//title//text() | //TITLE//text()').getall()
        if title_tag:
            title_text = " ".join(title_tag).strip()
            # Clean up common formatting issues in title tags
            title_text = " ".join(title_text.split())
            if title_text:
                return title_text
        
        # Fallback - try first heading of any type (both cases)
        title = selector.xpath('(//*[self::h1 or self::h2 or self::h3 or self::H1 or self::H2 or self::H3])[1]/text()').get()
        if title:
            return title.strip()
        
        return ""
    
    def parse_authors(self, html: str) -> List[str]:
        """Parse author information from MMWR HTML."""
        selector = Selector(text=html)
        authors = []
        
        # Try to get authors from meta tags first
        authors = self.get_authors_from_meta(selector)
        if authors:
            return authors
        
        # For MMWR, check for "Reported by:" pattern
        reported_by_nodes = selector.xpath("//p[contains(., 'Reported by:')]")
        for node in reported_by_nodes:
            text = "".join(node.xpath(".//text()").getall())
            if 'Reported by:' in text:
                # Extract text after "Reported by:"
                author_text = text.split("Reported by:", 1)[1]
                
                # Split on common delimiters
                raw_authors = re.split(r'[,;]|\band\b|\bfrom\b|\bat\b', author_text)
                
                # Clean each author name
                cleaned_authors = []
                for author in raw_authors:
                    clean_name = self.clean_author_name(author)
                    if clean_name and len(clean_name.split()) >= 2:  # Ensure at least first and last name
                        cleaned_authors.append(clean_name)
                
                if cleaned_authors:
                    authors.extend(cleaned_authors)
                    break
        
        # If no authors found via "Reported by:", try standard formats
        if not authors:
            # Try h4 with author class
            author_tags = selector.xpath("//h4[contains(@class, 'author')]//text()").getall()
            if author_tags:
                raw_authors = []
                current_author = []
                for tag in author_tags:
                    if tag.strip():
                        if ',' in tag or 'and' in tag.lower():
                            # Split on delimiters and add to raw_authors
                            parts = re.split(r'[,;]|\band\b', tag)
                            raw_authors.extend(parts)
                        else:
                            current_author.append(tag)
                
                if current_author:
                    raw_authors.append(' '.join(current_author))
                
                # Clean each author name
                for author in raw_authors:
                    clean_name = self.clean_author_name(author)
                    if clean_name and len(clean_name.split()) >= 2:
                        authors.append(clean_name)
        
        return authors
    
    def parse_abstract(self, html: str) -> str:
        """Parse the article abstract from MMWR HTML."""
        selector = Selector(text=html)
        
        # Try h2 with Abstract (common in MMWR)
        abstract_nodes = selector.xpath("//h2[contains(., 'Abstract')]/following-sibling::p//text()").getall()
        
        # If not found, try h3 with Abstract
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//h3[contains(., 'Abstract')]/following-sibling::p//text()").getall()
        
        # If still not found, try div or section with abstract class
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//div[contains(@class, 'abstract')]//p//text()").getall()
        
        # Fallback: Look for paragraphs with "Summary-Abstract-Text" class (common in MMWR preview articles)
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//p[contains(@class, 'Summary-Abstract-Text')]//text()").getall()
        
        # Additional fallback: Look for abstract content after "Abstract" title paragraph
        if not abstract_nodes:
            abstract_title = selector.xpath("//p[contains(@class, 'Summary-Abstract-Title')]")
            if abstract_title:
                # Get all following paragraphs with Summary-Abstract-Text class
                following_abstract = selector.xpath("//p[contains(@class, 'Summary-Abstract-Title')]/following-sibling::p[contains(@class, 'Summary-Abstract-Text')]//text()").getall()
                if following_abstract:
                    abstract_nodes = following_abstract
        
        if abstract_nodes:
            abstract = " ".join(abstract_nodes)
            return self.clean_text_with_notices(abstract)
        
        return ""


class PCDArticleParser(CDCArticleParser):
    """Parser for PCD (Preventing Chronic Disease) articles."""
    
    def parse_title(self, html: str) -> str:
        """Parse the article title from PCD HTML."""
        selector = Selector(text=html)
        
        # Try to find title in h1 tags (common in PCD)
        title = selector.xpath('//h1/text()').get()
        if title and title.strip():
            return title.strip()
        
        # Fallback - try first heading of any type
        title = selector.xpath('(//*[self::h1 or self::h2 or self::h3])[1]/text()').get()
        if title and title.strip():
            return title.strip()
        
        # Try to get text after span in h1 (for older PCD articles)
        title_after_span = selector.xpath('//h1/span/following-sibling::text()').getall()
        if title_after_span:
            title = ' '.join(title_after_span).strip()
            # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
            title = ' '.join(title.split())
            if title:
                return title
        
        # Final fallback - use normalize-space to get all text content from headings
        title = selector.xpath('normalize-space(//h1)').get()
        if title and title.strip():
            return title.strip()
        
        title = selector.xpath('normalize-space((//*[self::h1 or self::h2 or self::h3])[1])').get()
        if title and title.strip():
            return title.strip()
        
        return ""
    
    def parse_authors(self, html: str) -> List[str]:
        """Parse author information from PCD HTML."""
        selector = Selector(text=html)
        authors = []
        
        # Try to get authors from meta tags first (common in PCD)
        authors = self.get_authors_from_meta(selector)
        if authors:
            return authors
        
        # Try h4 with author class
        author_tags = selector.xpath("//h4[contains(@class, 'author')]//text()").getall()
        if author_tags:
            raw_authors = []
            current_author = []
            for tag in author_tags:
                if tag.strip():
                    if ',' in tag or 'and' in tag.lower():
                        # Split on delimiters and add to raw_authors
                        parts = re.split(r'[,;]|\band\b', tag)
                        raw_authors.extend(parts)
                    else:
                        current_author.append(tag)
            
            if current_author:
                raw_authors.append(' '.join(current_author))
            
            # Clean each author name
            for author in raw_authors:
                clean_name = self.clean_author_name(author)
                if clean_name and len(clean_name.split()) >= 2:
                    authors.append(clean_name)
        
        # Fallback - try h4 without class requirement
        if not authors:
            author_tags = selector.xpath("//h4//text()").getall()
            if author_tags:
                raw_authors = []
                current_author = []
                for tag in author_tags:
                    if tag.strip():
                        if ',' in tag or 'and' in tag.lower():
                            # Split on delimiters and add to raw_authors
                            parts = re.split(r'[,;]|\band\b', tag)
                            raw_authors.extend(parts)
                        else:
                            current_author.append(tag)
                
                if current_author:
                    raw_authors.append(' '.join(current_author))
                
                # Clean each author name
                for author in raw_authors:
                    clean_name = self.clean_author_name(author)
                    if clean_name and len(clean_name.split()) >= 2:
                        authors.append(clean_name)
        
        return authors
    
    def parse_abstract(self, html: str) -> str:
        """Parse the article abstract from PCD HTML."""
        selector = Selector(text=html)
        
        # Try h2 with Abstract (common in PCD) - check both exact text and contains
        abstract_h2 = selector.xpath("//h2[text()='Abstract']/text()").get()
        if not abstract_h2:
            abstract_h2 = selector.xpath("//h2[normalize-space(.)='Abstract']").get()
        if abstract_h2:
            # Get all paragraphs following Abstract h2 (excluding small text paragraphs)
            all_ps = selector.xpath("//h2[text()='Abstract' or normalize-space(.)='Abstract']/following-sibling::p[not(contains(@class,'psmall'))]")
            results = []
            for p in all_ps:
                # Get the section this paragraph belongs to (the preceding h2)
                section_node = p.xpath(".//preceding-sibling::h2[1]").get()
                if section_node:
                    # Use normalize-space to get full text including nested elements
                    section_selector = Selector(text=section_node)
                    section = section_selector.xpath("normalize-space(.)").get()
                    if section:
                        section = section.strip().lower()
                        data = " ".join(p.xpath(".//text()").getall()).replace("\n", "")
                        results.append({"section": section, "data": data})
            
            # Extract only paragraphs that belong to the abstract section
            extract_section = lambda x, rs: " ".join([ele['data'] for ele in rs if ele['section'] == x])
            abstract_text = extract_section('abstract', results)
            
            if abstract_text:
                return self.clean_text_with_notices(abstract_text)
        
        
        # If not found, try h3 with Abstract
        abstract_nodes = selector.xpath("//h3[contains(., 'Abstract')]/following-sibling::p//text()").getall()
        if not abstract_nodes:
            abstract_nodes = selector.xpath("//div[contains(@class, 'abstract')]//p//text()").getall()
        
        if abstract_nodes:
            abstract = " ".join(abstract_nodes)
            return self.clean_text_with_notices(abstract)
        
        return ""


# Factory function to create appropriate parser
def create_parser(collection: str, journal_type: str, language: str, articles_collection: dict, validate_articles: bool = True) -> CDCArticleParser:
    """Create appropriate parser based on collection type."""
    collection_lower = collection.lower()
    
    if collection_lower == 'eid':
        return EIDArticleParser(collection, journal_type, language, articles_collection, validate_articles)
    elif collection_lower == 'mmwr':
        return MMWRArticleParser(collection, journal_type, language, articles_collection, validate_articles)
    elif collection_lower == 'pcd':
        return PCDArticleParser(collection, journal_type, language, articles_collection, validate_articles)
    else:
        raise ValueError(f"Unknown collection: {collection}. Supported collections: eid, mmwr, pcd")


if __name__ == '__main__':
    #f = '/Users/mayerantoine/Code/cdc-text-corpora/cdc-corpus-data/json-html/pcd/issues/2005/nov/05_0059.htm'
    #f = '/Users/mayerantoine/Code/cdc-text-corpora/cdc-corpus-data/json-html/pcd/issues/2019/18_0093.htm'
    f = '/Users/mayerantoine/Code/cdc-text-corpora/notebooks/cdc-corpus-data/json-html/mmwr/preview/mmwrhtml/00050906.htm'
    loader = HTMLArticleLoader('mmwr','','en')
    html_data = loader._load_html_file(f)
    #print(data)
    
    parser = create_parser('mmwr','','en',{'1':html_data})
    data = parser.parse_article(relative_url='', html=html_data)
    print(data)
    #title = parser.parse_title(data)
    #abstract = parser.parse_abstract(data)
    #print(title)
    #print(abstract)