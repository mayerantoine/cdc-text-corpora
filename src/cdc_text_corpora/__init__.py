"""CDC Text Corpora package for easy access to PCD, EID, and MMWR collections."""

from .core.datasets import CDCCorpus
from .core.parser import Article, CDCCollections
from .index import ArticleIndexer, IndexConfig

__version__ = "0.1.0"
__all__ = ["CDCCorpus", "Article", "CDCCollections", "ArticleIndexer", "IndexConfig", "main"]


def main() -> None:
    """Main entry point for the cdc-text-corpora CLI."""
    from .cli.main import main as cli_main
    cli_main()
