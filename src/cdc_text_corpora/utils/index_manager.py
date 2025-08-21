"""Index management utilities for CDC Text Corpora.

This module handles the extraction and management of pre-built ChromaDB indexes
and provides utilities for detecting available HTML articles.
"""

import tarfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.utils.config import get_data_directory


class IndexManager:
    """
    Manages ChromaDB index extraction and HTML article detection.
    
    This class handles:
    - Detection of existing HTML articles in cdc-corpus-data/json-html/
    - Extraction of bundled pre-built ChromaDB index
    - Validation of index integrity
    - User guidance for data setup
    """
    
    def __init__(self, corpus_manager: CDCCorpus, console: Optional[Console] = None):
        """
        Initialize IndexManager.
        
        Args:
            corpus_manager: CDCCorpus instance for accessing data directories
            console: Optional Rich console for output (creates new if None)
        """
        self.corpus_manager = corpus_manager
        self.console = console or Console()
        self.data_directory = corpus_manager.get_data_directory()
        self.chroma_db_path = self.data_directory / "chroma_db"
        
    def has_html_articles(self, collection: Optional[str] = None) -> bool:
        """
        Check if HTML articles exist in cdc-corpus-data/json-html/.
        
        Args:
            collection: Optional collection filter (pcd, eid, mmwr)
            
        Returns:
            True if HTML articles are found, False otherwise
        """
        html_dir = self.data_directory / "json-html"
        
        if not html_dir.exists():
            return False
            
        if collection:
            # Check specific collection directory
            collection_dir = html_dir / collection.lower()
            if not collection_dir.exists():
                return False
            return any(collection_dir.rglob("*.htm"))
        else:
            # Check any collection
            return any(html_dir.rglob("*.htm"))
    
    def get_html_articles_count(self, collection: Optional[str] = None) -> Dict[str, int]:
        """
        Get count of HTML articles by collection.
        
        Args:
            collection: Optional collection filter
            
        Returns:
            Dictionary with collection counts
        """
        html_dir = self.data_directory / "json-html"
        counts = {}
        
        if not html_dir.exists():
            return counts
            
        collections_to_check = [collection.lower()] if collection else ['pcd', 'eid', 'mmwr']
        
        for coll in collections_to_check:
            coll_dir = html_dir / coll
            if coll_dir.exists():
                counts[coll] = len(list(coll_dir.rglob("*.htm")))
            else:
                counts[coll] = 0
                
        return counts
    
    def has_existing_index(self) -> bool:
        """
        Check if a valid ChromaDB index already exists.
        
        Returns:
            True if valid index exists, False otherwise
        """
        if not self.chroma_db_path.exists():
            return False
            
        # Check for key ChromaDB files
        sqlite_file = self.chroma_db_path / "chroma.sqlite3"
        return sqlite_file.exists() and sqlite_file.stat().st_size > 0
    
    def get_bundled_index_path(self) -> Optional[Path]:
        """
        Get path to bundled chroma_index.tar.xz file.
        
        Returns:
            Path to bundled index file if it exists, None otherwise
        """
        # Priority order for finding bundled index:
        # 1. In user's data directory (after installation)
        bundled_path = self.data_directory / "chroma_index.tar.xz"
        if bundled_path.exists():
            return bundled_path
            
        # 2. In current working directory (development mode)
        cwd_path = Path.cwd() / "cdc-corpus-data" / "chroma_index.tar.xz"
        if cwd_path.exists():
            return cwd_path
            
        # 3. Try to find in package installation (wheel/sdist)
        try:
            import cdc_text_corpora
            package_dir = Path(cdc_text_corpora.__file__).parent
            package_data_path = package_dir / "data" / "chroma_index.tar.xz"
            if package_data_path.exists():
                return package_data_path
        except (ImportError, AttributeError):
            pass
            
        return None
    
    def extract_bundled_index(self, force: bool = False) -> bool:
        """
        Extract the bundled ChromaDB index to cdc-corpus-data/chroma_db/.
        
        Args:
            force: If True, overwrite existing index
            
        Returns:
            True if extraction successful, False otherwise
        """
        bundled_path = self.get_bundled_index_path()
        
        if not bundled_path:
            self.console.print("[red]❌ Bundled index file not found[/red]")
            return False
            
        if self.has_existing_index() and not force:
            self.console.print("[yellow]⚠️  ChromaDB index already exists. Use force=True to overwrite.[/yellow]")
            return False
            
        try:
            # Check available disk space
            import shutil as disk_utils
            free_space = disk_utils.disk_usage(self.data_directory).free
            archive_size = bundled_path.stat().st_size
            # Assume 3x expansion ratio (typical for compressed data)
            estimated_extracted_size = archive_size * 3
            
            if free_space < estimated_extracted_size:
                self.console.print(f"[red]❌ Insufficient disk space[/red]")
                self.console.print(f"Required: ~{estimated_extracted_size // (1024**3):.1f}GB, Available: {free_space // (1024**3):.1f}GB")
                return False
            
            # Remove existing index if forcing
            if force and self.chroma_db_path.exists():
                self.console.print("[yellow]🗑️  Removing existing index...[/yellow]")
                shutil.rmtree(self.chroma_db_path)
                
            # Create directory
            self.chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            # Verify archive integrity before extraction
            try:
                with tarfile.open(bundled_path, 'r:xz') as tar:
                    # Just check if we can read the header
                    members = tar.getmembers()
                    if not members:
                        self.console.print("[red]❌ Bundled index archive appears to be empty[/red]")
                        return False
            except tarfile.TarError as e:
                self.console.print(f"[red]❌ Bundled index archive is corrupted: {e}[/red]")
                return False
            
            # Extract with progress bar
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Extracting bundled index...", total=100)
                
                with tarfile.open(bundled_path, 'r:xz') as tar:
                    # Extract all members
                    members = tar.getmembers()
                    total_members = len(members)
                    
                    for i, member in enumerate(members):
                        try:
                            tar.extract(member, self.data_directory)
                            progress.update(task, completed=(i + 1) * 100 // total_members)
                        except (OSError, tarfile.ExtractError) as e:
                            self.console.print(f"[red]❌ Failed to extract {member.name}: {e}[/red]")
                            return False
            
            # Validate extraction
            if self.has_existing_index():
                self.console.print("[green]✅ Bundled index extracted successfully[/green]")
                return True
            else:
                self.console.print("[red]❌ Index extraction failed - validation failed[/red]")
                self.console.print("[yellow]💡 Try downloading and parsing articles manually:[/yellow]")
                self.console.print("   cdc-corpus download --collection pcd")
                self.console.print("   cdc-corpus parse --collection pcd")
                return False
                
        except PermissionError:
            self.console.print(f"[red]❌ Permission denied writing to {self.data_directory}[/red]")
            self.console.print("[yellow]💡 Try running with appropriate permissions or in a different directory[/yellow]")
            return False
        except OSError as e:
            self.console.print(f"[red]❌ Disk I/O error during extraction: {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Unexpected error during index extraction: {e}[/red]")
            self.console.print(f"[yellow]💡 Please report this issue with the error details above[/yellow]")
            return False
    
    def get_index_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of index and data availability.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "has_existing_index": self.has_existing_index(),
            "has_html_articles": self.has_html_articles(),
            "has_bundled_index": self.get_bundled_index_path() is not None,
            "html_counts": self.get_html_articles_count(),
            "data_directory": str(self.data_directory),
            "chroma_db_path": str(self.chroma_db_path),
            "recommendations": []
        }
        
        # Generate recommendations
        if status["has_existing_index"]:
            status["recommendations"].append("✅ Ready to use - existing ChromaDB index found")
        elif status["has_html_articles"]:
            status["recommendations"].append("📚 HTML articles found - can build fresh index")
        elif status["has_bundled_index"]:
            status["recommendations"].append("📦 Bundled index available - can extract for immediate use")
        else:
            status["recommendations"].append("⬇️  Download data first: cdc-corpus download --collection <name>")
            
        return status
    
    def show_guidance(self) -> None:
        """Show helpful guidance based on current data availability."""
        status = self.get_index_status()
        
        self.console.print("\n[bold cyan]📊 Data & Index Status[/bold cyan]")
        self.console.print(f"Data Directory: {status['data_directory']}")
        self.console.print(f"ChromaDB Index: {'✅ Exists' if status['has_existing_index'] else '❌ Missing'}")
        self.console.print(f"HTML Articles: {'✅ Found' if status['has_html_articles'] else '❌ Missing'}")
        self.console.print(f"Bundled Index: {'✅ Available' if status['has_bundled_index'] else '❌ Not Available'}")
        
        if status['html_counts']:
            self.console.print("\n[bold cyan]HTML Articles by Collection:[/bold cyan]")
            for collection, count in status['html_counts'].items():
                if count > 0:
                    self.console.print(f"  {collection.upper()}: {count:,} articles")
        
        self.console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")
        for rec in status['recommendations']:
            self.console.print(f"  {rec}")
        
        if not status['has_existing_index'] and not status['has_html_articles']:
            self.console.print("\n[bold yellow]🚀 Quick Start Options:[/bold yellow]")
            if status['has_bundled_index']:
                self.console.print("  1. Use bundled index: [bold]cdc-corpus index --use-bundled[/bold]")
                self.console.print("  2. Download fresh data: [bold]cdc-corpus download --collection pcd[/bold]")
            else:
                self.console.print("  1. Download data: [bold]cdc-corpus download --collection pcd[/bold]")
                self.console.print("  2. Parse articles: [bold]cdc-corpus parse --collection pcd[/bold]")