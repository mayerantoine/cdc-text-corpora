"""Main CLI entry point for cdc-text-corpora."""

import typer
from rich.console import Console
from rich.panel import Panel
import pathlib
from cdc_text_corpora.core.downloader import download_collection
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.utils.config import get_data_directory


app = typer.Typer(
    name="cdc-corpus",
    help="CDC Text Corpora: Access to PCD, EID, and MMWR collections with semantic search and QA",
)

console = Console()


@app.command()
def download(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c",
        help="Collection to download: pcd, eid, mmwr, metadata, or all",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
):
    """Download CDC text corpora collections."""
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "metadata", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Display collection info
    collection_info = {
        "pcd": "Preventing Chronic Disease",
        "eid": "Emerging Infectious Diseases", 
        "mmwr": "Morbidity and Mortality Weekly Report",
        "metadata": "Collection metadata only",
        "all": "All collections and metadata"
    }
    
    collection_lower = collection.lower()
    
    console.print(Panel(
        f"[bold blue]Downloading: {collection_info[collection_lower]}[/bold blue]",
        title="CDC Text Corpora Downloader",
        border_style="blue"
    ))
    
    try:
        # Call the existing download function - it has its own rich progress bars
        download_collection(collection_lower)
        
        console.print(f"[green]✓ Successfully downloaded {collection_lower} collection[/green]")
        
        if verbose:
            # Get the actual data directory path
            data_path = get_data_directory()
            console.print(f"[dim]Files saved to: {data_path}[/dim]")
            
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)



@app.command()
def parse(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c",
        help="Collection to parse: pcd, eid, mmwr, or all",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        "-l",
        help="Language to parse: en, es, fr, zhs, zht, or all",
    ),
    save_json: bool = typer.Option(
        False,
        "--save-json",
        "-j",
        help="Save parsed articles as JSON files",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory for JSON files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
):
    """Parse CDC text corpora collections into structured data."""
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate language input
    valid_languages = ["en", "es", "fr", "zhs", "zht", "all"]
    if language.lower() not in valid_languages:
        console.print(f"[red]Error: Invalid language '{language}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_languages)}[/yellow]")
        raise typer.Exit(1)
    
    # Display parsing info
    collection_info = {
        "pcd": "Preventing Chronic Disease",
        "eid": "Emerging Infectious Diseases", 
        "mmwr": "Morbidity and Mortality Weekly Report",
        "all": "All collections"
    }
    
    language_info = {
        "en": "English",
        "es": "Spanish",
        "fr": "French", 
        "zhs": "Simplified Chinese",
        "zht": "Traditional Chinese",
        "all": "All languages"
    }
    
    collection_lower = collection.lower()
    language_lower = language.lower() if language.lower() != "all" else None
    
    console.print(Panel(
        f"[bold blue]Parsing: {collection_info[collection_lower]} ({language_info[language.lower()]})[/bold blue]\n"
        f"[dim]Save JSON: {'Yes' if save_json else 'No'}[/dim]",
        title="CDC Text Corpora Parser",
        border_style="blue"
    ))
    
    try:
        # Initialize corpus manager
        corpus = CDCCorpus()
        
        # Determine collections to process
        collections_to_process = ['pcd', 'eid', 'mmwr'] if collection_lower == 'all' else [collection_lower]
        
        total_parsed = 0
        json_files_created = []
        
        for coll in collections_to_process:
            console.print(f"\n[bold cyan]Processing {coll.upper()} collection...[/bold cyan]")
            
            # Check if collection is downloaded
            if not corpus.is_collection_downloaded(coll):
                console.print(f"[yellow]Warning: {coll.upper()} collection not downloaded. Skipping...[/yellow]")
                console.print(f"[dim]Download with: cdc-corpus download -c {coll}[/dim]")
                continue
            
            # Parse the collection
            result = corpus.load_collection_articles(
                collection=coll,
                language=language_lower,
                save_json=save_json,
                output_dir=output_dir
            )
            
            # Check for errors
            if 'error' in result['stats']:
                console.print(f"[red]Error: {result['stats']['error']}[/red]")
                continue
            
            # Display results
            parsed_count = result['stats']['parsed_count']
            total_parsed += parsed_count
            
            console.print(f"[green]✓ Parsed {parsed_count} articles from {coll.upper()}[/green]")
            
            if save_json and 'json_file_path' in result:
                json_files_created.append(result['json_file_path'])
                console.print(f"[green]✓ Saved to: {result['json_file_path']}[/green]")
            
            if verbose:
                console.print(f"[dim]  HTML files loaded: {result['stats']['html_count']}[/dim]")
                console.print(f"[dim]  Language: {result['stats']['language']}[/dim]")
        
        # Final summary
        console.print(f"\n[bold green]✅ Parsing complete![/bold green]")
        console.print(f"[green]Total articles parsed: {total_parsed}[/green]")
        
        if json_files_created:
            console.print(f"[green]JSON files created: {len(json_files_created)}[/green]")
            if verbose:
                for json_file in json_files_created:
                    console.print(f"[dim]  {json_file}[/dim]")
        
        if verbose:
            data_path = get_data_directory()
            console.print(f"[dim]Data directory: {data_path}[/dim]")
            
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def qa():
    """QA CDC text corpora collections."""
    print(f"RAG code")

@app.command()
def search():
    """Search CDC text corpora collections."""
    print(f"RAG code")

def main():
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()