"""Article validation utilities for CDC Text Corpora."""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse

from cdc_text_corpora.core.parser import Article, CDCCollections


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Critical issues that make article unusable
    WARNING = "warning"  # Issues that may affect quality but don't break functionality
    INFO = "info"       # Minor issues or suggestions


@dataclass
class ValidationIssue:
    """Represents a validation issue found in an article."""
    field: str
    severity: ValidationSeverity
    message: str
    value: Optional[str] = None


@dataclass
class ValidationResult:
    """Results of article validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]


class ArticleValidator:
    """Validates parsed CDC articles against quality requirements."""
    
    def __init__(self) -> None:
        self.cdc_url_pattern = re.compile(
            r'^https?://(?:www\.)?cdc\.gov/',
            re.IGNORECASE
        )
    
    def validate_article(self, article: Article) -> ValidationResult:
        """Validate a single article against all rules."""
        issues = []
        

        # Run all validation checks
        issues.extend(self._validate_title(article))
        issues.extend(self._validate_urls(article))
        issues.extend(self._validate_url_format(article))
        issues.extend(self._validate_authors(article))
        issues.extend(self._validate_full_text(article))
        issues.extend(self._validate_html_text(article))
        issues.extend(self._validate_collection(article))
        issues.extend(self._validate_language(article))
        
        # Article is valid if there are no ERROR-level issues
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def _validate_title(self, article: Article) -> List[ValidationIssue]:
        """Validate article title requirements."""
        issues = []
        
        if not article.title:
            issues.append(ValidationIssue(
                field="title",
                severity=ValidationSeverity.ERROR,
                message="Article must have a non-empty title",
                value=article.title
            ))
        elif len(article.title.strip()) < 5:
            issues.append(ValidationIssue(
                field="title",
                severity=ValidationSeverity.WARNING,
                message="Article title is very short (less than 5 characters)",
                value=article.title
            ))
        elif len(article.title) > 300:
            issues.append(ValidationIssue(
                field="title",
                severity=ValidationSeverity.WARNING,
                message="Article title is very long (over 300 characters)",
                value=f"{article.title[:50]}..."
            ))
        
        return issues
    
    def _validate_urls(self, article: Article) -> List[ValidationIssue]:
        """Validate URL requirements."""
        issues = []
        
      
        if not article.url:
            issues.append(ValidationIssue(
                field="url",
                severity=ValidationSeverity.ERROR,
                message="Article must have a URL",
                value=article.url
            ))
        
        if not article.relative_url or not article.relative_url.strip():
            print("not FOUND:",article.relative_url)
            issues.append(ValidationIssue(
                field="relative_url",
                severity=ValidationSeverity.ERROR,
                message="Article must have a relative URL",
                value=article.relative_url
            ))
        
        return issues
    
    def _validate_url_format(self, article: Article) -> List[ValidationIssue]:
        """Validate URL format requirements."""
        issues = []
        
        if article.url:
            # Check if URL is a valid CDC URL
            if not self.cdc_url_pattern.match(article.url):
                issues.append(ValidationIssue(
                    field="url",
                    severity=ValidationSeverity.WARNING,
                    message="Article URL should be a valid CDC domain URL",
                    value=article.url
                ))
            
            # Check if URL is properly formatted
            try:
                parsed = urlparse(article.url)
                if not parsed.scheme or not parsed.netloc:
                    issues.append(ValidationIssue(
                        field="url",
                        severity=ValidationSeverity.ERROR,
                        message="Article URL is not properly formatted",
                        value=article.url
                    ))
            except Exception:
                issues.append(ValidationIssue(
                    field="url",
                    severity=ValidationSeverity.ERROR,
                    message="Article URL cannot be parsed",
                    value=article.url
                ))
        
        return issues
    
    def _validate_authors(self, article: Article) -> List[ValidationIssue]:
        """Validate author requirements."""
        issues = []
        
        if not article.authors or len(article.authors) == 0:
            issues.append(ValidationIssue(
                field="authors",
                severity=ValidationSeverity.ERROR,
                message="Article must have at least one author",
                value=str(article.authors)
            ))
        else:
            # Check for empty author names
            empty_authors = [i for i, author in enumerate(article.authors) if not author.strip()]
            if empty_authors:
                issues.append(ValidationIssue(
                    field="authors",
                    severity=ValidationSeverity.WARNING,
                    message=f"Empty author names found at positions: {empty_authors}",
                    value=str(article.authors)
                ))
            
            # Check for very short author names (likely parsing errors)
            short_authors = [author for author in article.authors if len(author.strip()) < 3]
            if short_authors:
                issues.append(ValidationIssue(
                    field="authors",
                    severity=ValidationSeverity.WARNING,
                    message=f"Very short author names found: {short_authors}",
                    value=str(article.authors)
                ))
        
        return issues
    
    def _validate_full_text(self, article: Article) -> List[ValidationIssue]:
        """Validate full_text content requirements."""
        issues = []
        
        if not article.full_text:
            issues.append(ValidationIssue(
                field="full_text",
                severity=ValidationSeverity.ERROR,
                message="Article must have full_text content",
                value=f"Length: {len(article.full_text) if article.full_text else 0}"
            ))
        elif len(article.full_text.strip()) < 100:
            issues.append(ValidationIssue(
                field="full_text",
                severity=ValidationSeverity.WARNING,
                message="Article full_text is very short (less than 100 characters)",
                value=f"Length: {len(article.full_text)}"
            ))
        
        return issues
    
    def _validate_html_text(self, article: Article) -> List[ValidationIssue]:
        """Validate html_text content requirements."""
        issues = []
        
        if not article.html_text:
            issues.append(ValidationIssue(
                field="html_text",
                severity=ValidationSeverity.ERROR,
                message="Article must have html_text content",
                value=f"Length: {len(article.html_text) if article.html_text else 0}"
            ))
        elif len(article.html_text.strip()) < 200:
            issues.append(ValidationIssue(
                field="html_text",
                severity=ValidationSeverity.WARNING,
                message="Article html_text is very short (less than 200 characters)",
                value=f"Length: {len(article.html_text)}"
            ))
        
        return issues
    
    def _validate_collection(self, article: Article) -> List[ValidationIssue]:
        """Validate collection requirements."""
        issues = []
        
        if not article.collection:
            issues.append(ValidationIssue(
                field="collection",
                severity=ValidationSeverity.ERROR,
                message="Article must have a collection assigned",
                value=str(article.collection)
            ))
        elif article.collection not in CDCCollections:
            issues.append(ValidationIssue(
                field="collection",
                severity=ValidationSeverity.ERROR,
                message="Article collection must be a valid CDCCollections enum value",
                value=str(article.collection)
            ))
        
        return issues
    
    def _validate_language(self, article: Article) -> List[ValidationIssue]:
        """Validate language requirements."""
        issues = []
        
        if not article.language:
            issues.append(ValidationIssue(
                field="language",
                severity=ValidationSeverity.WARNING,
                message="Article should have a language specified",
                value=article.language
            ))
        elif article.language not in ['en', 'es', 'fr', 'zhs', 'zht']:
            issues.append(ValidationIssue(
                field="language",
                severity=ValidationSeverity.WARNING,
                message="Article language should be one of: en, es, fr, zhs, zht",
                value=article.language
            ))
        
        return issues


class BatchValidationReport:
    """Generates validation reports for multiple articles."""
    
    def __init__(self, validator: ArticleValidator):
        self.validator = validator
    
    def validate_articles(self, articles: Dict[str, Article]) -> Dict[str, ValidationResult]:
        """Validate multiple articles and return results."""
        results = {}
        
        for article_id, article in articles.items():
            results[article_id] = self.validator.validate_article(article)
        
        return results
    
    def generate_summary_report(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate a summary report of validation results."""
        total_articles = len(results)
        valid_articles = sum(1 for result in results.values() if result.is_valid)
        
        # Count issues by severity
        error_count = sum(len(result.get_issues_by_severity(ValidationSeverity.ERROR)) 
                         for result in results.values())
        warning_count = sum(len(result.get_issues_by_severity(ValidationSeverity.WARNING)) 
                           for result in results.values())
        info_count = sum(len(result.get_issues_by_severity(ValidationSeverity.INFO)) 
                        for result in results.values())
        
        # Count articles with issues
        articles_with_errors = sum(1 for result in results.values() if result.has_errors)
        articles_with_warnings = sum(1 for result in results.values() if result.has_warnings)
        
        # Field-specific issue counts
        field_issues: Dict[str, int] = {}
        for result in results.values():
            for issue in result.issues:
                if issue.field not in field_issues:
                    field_issues[issue.field] = 0
                field_issues[issue.field] += 1
        
        return {
            "total_articles": total_articles,
            "valid_articles": valid_articles,
            "invalid_articles": total_articles - valid_articles,
            "validation_rate": (valid_articles / total_articles) * 100 if total_articles > 0 else 0,
            "total_issues": error_count + warning_count + info_count,
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "articles_with_errors": articles_with_errors,
            "articles_with_warnings": articles_with_warnings,
            "field_issues": field_issues
        }
    
    def print_summary_report(self, results: Dict[str, ValidationResult]) -> None:
        """Print a formatted summary report."""
        summary = self.generate_summary_report(results)
        
        print("\n" + "="*60)
        print("ARTICLE VALIDATION SUMMARY REPORT")
        print("="*60)
        
        print(f"Total Articles: {summary['total_articles']}")
        print(f"Valid Articles: {summary['valid_articles']}")
        print(f"Invalid Articles: {summary['invalid_articles']}")
        print(f"Validation Rate: {summary['validation_rate']:.1f}%")
        
        print(f"\nIssue Breakdown:")
        print(f"  Errors: {summary['errors']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Info: {summary['info']}")
        
        print(f"\nArticles with Issues:")
        print(f"  With Errors: {summary['articles_with_errors']}")
        print(f"  With Warnings: {summary['articles_with_warnings']}")
        
        if summary['field_issues']:
            print(f"\nMost Common Field Issues:")
            sorted_fields = sorted(summary['field_issues'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for field, count in sorted_fields[:5]:
                print(f"  {field}: {count} issues")
        
        print("="*60)
    
    def print_detailed_issues(self, results: Dict[str, ValidationResult], 
                            severity_filter: Optional[ValidationSeverity] = None,
                            max_articles: int = 10) -> None:
        """Print detailed validation issues."""
        print(f"\nDETAILED VALIDATION ISSUES")
        print("-"*50)
        
        count = 0
        for article_id, result in results.items():
            if count >= max_articles:
                break
                
            issues_to_show = result.issues
            if severity_filter:
                issues_to_show = result.get_issues_by_severity(severity_filter)
            
            if issues_to_show:
                print(f"\nArticle: {article_id}")
                for issue in issues_to_show:
                    print(f"  [{issue.severity.value.upper()}] {issue.field}: {issue.message}")
                    if issue.value:
                        print(f"    Value: {issue.value}")
                count += 1