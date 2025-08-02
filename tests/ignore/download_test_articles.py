#!/usr/bin/env python3
"""
Script to download HTML articles from URLs listed in articles_test.md
and save them as .htm files in tests/articles/ directory.
"""

import os
import sys
import re
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


def read_test_urls(test_md_path: str) -> List[str]:
    """Read URLs from the articles_test.md file.
    
    Args:
        test_md_path: Path to the articles_test.md file
        
    Returns:
        List of URLs to download
    """
    urls = []
    
    try:
        with open(test_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract URLs using regex - look for lines starting with https://
        url_pattern = r'^https://[^\s]+$'
        for line in content.split('\n'):
            line = line.strip()
            if re.match(url_pattern, line):
                urls.append(line)
        
        console.print(f"[green]Found {len(urls)} URLs in {test_md_path}[/green]")
        return urls
        
    except Exception as e:
        console.print(f"[red]Error reading {test_md_path}: {e}[/red]")
        return []


def generate_filename_from_url(url: str) -> str:
    """Generate a safe filename from a URL.
    
    Args:
        url: The URL to convert
        
    Returns:
        Safe filename for the HTML file
    """
    # Parse the URL
    parsed = urlparse(url)
    
    # Extract path and remove leading slash
    path = parsed.path.lstrip('/')
    
    # Replace path separators with underscores
    filename = path.replace('/', '_')
    
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    
    # Ensure it ends with .htm
    if not filename.endswith('.htm') and not filename.endswith('.html'):
        filename += '.htm'
    
    # If filename is too long, truncate but keep the extension
    if len(filename) > 200:
        name_part = filename.rsplit('.', 1)[0][:190]
        ext_part = filename.rsplit('.', 1)[1]
        filename = f"{name_part}.{ext_part}"
    
    return filename


def download_html_content(url: str, timeout: int = 30) -> str:
    """Download HTML content from a URL.
    
    Args:
        url: URL to download
        timeout: Request timeout in seconds
        
    Returns:
        HTML content as string
        
    Raises:
        Exception: If download fails
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    return response.text


def download_test_articles(test_md_path: str, output_dir: str) -> dict:
    """Download all test articles from URLs in the markdown file.
    
    Args:
        test_md_path: Path to articles_test.md
        output_dir: Directory to save downloaded HTML files
        
    Returns:
        Dictionary with download statistics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read URLs from markdown file
    urls = read_test_urls(test_md_path)
    
    if not urls:
        console.print("[yellow]No URLs found to download[/yellow]")
        return {"total": 0, "successful": 0, "failed": 0, "files": []}
    
    console.print(f"[cyan]Downloading {len(urls)} articles to {output_path}[/cyan]")
    
    successful_downloads = []
    failed_downloads = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Downloading articles...", total=len(urls))
        
        for i, url in enumerate(urls, 1):
            try:
                progress.update(task, description=f"Downloading {i}/{len(urls)}: {url[:50]}...")
                
                # Generate filename
                filename = generate_filename_from_url(url)
                file_path = output_path / filename
                
                # Skip if file already exists
                if file_path.exists():
                    console.print(f"[yellow]Skipping {filename} (already exists)[/yellow]")
                    successful_downloads.append(str(file_path))
                    progress.advance(task)
                    continue
                
                # Download HTML content
                html_content = download_html_content(url)
                
                # Save to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                successful_downloads.append(str(file_path))
                console.print(f"[green]✓ Downloaded: {filename}[/green]")
                
            except Exception as e:
                failed_downloads.append({"url": url, "error": str(e)})
                console.print(f"[red]✗ Failed to download {url}: {e}[/red]")
            
            progress.advance(task)
    
    # Print summary
    console.print(f"\n[bold green]Download Summary:[/bold green]")
    console.print(f"[green]Total URLs: {len(urls)}[/green]")
    console.print(f"[green]Successful: {len(successful_downloads)}[/green]")
    console.print(f"[red]Failed: {len(failed_downloads)}[/red]")
    
    if failed_downloads:
        console.print(f"\n[red]Failed Downloads:[/red]")
        for failure in failed_downloads:
            console.print(f"[red]  {failure['url']}: {failure['error']}[/red]")
    
    return {
        "total": len(urls),
        "successful": len(successful_downloads),
        "failed": len(failed_downloads),
        "files": successful_downloads,
        "failures": failed_downloads
    }


def create_download_manifest(stats: dict, output_dir: str) -> str:
    """Create a manifest file documenting the downloaded articles.
    
    Args:
        stats: Download statistics
        output_dir: Output directory
        
    Returns:
        Path to the created manifest file
    """
    import json
    from datetime import datetime
    
    manifest_path = Path(output_dir) / "download_manifest.json"
    
    manifest_data = {
        "created_at": datetime.now().isoformat(),
        "description": "Test HTML articles downloaded from specific CDC URLs",
        "download_stats": {
            "total_urls": stats["total"],
            "successful_downloads": stats["successful"],
            "failed_downloads": stats["failed"]
        },
        "downloaded_files": [str(Path(f).name) for f in stats["files"]],
        "failed_urls": [failure["url"] for failure in stats.get("failures", [])]
    }
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ Created download manifest: {manifest_path}[/green]")
    return str(manifest_path)


def main():
    """Main function to download test articles."""
    import argparse
    
    # Set up default paths
    script_dir = Path(__file__).parent
    default_test_md = script_dir / "articles_test.md"
    default_output_dir = script_dir / "articles"
    
    parser = argparse.ArgumentParser(
        description="Download HTML articles from URLs in articles_test.md"
    )
    parser.add_argument(
        "--test-md",
        type=str,
        default=str(default_test_md),
        help=f"Path to articles_test.md file (default: {default_test_md})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir),
        help=f"Output directory for downloaded HTML files (default: {default_output_dir})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Check if test markdown file exists
    if not Path(args.test_md).exists():
        console.print(f"[red]Error: Test markdown file not found: {args.test_md}[/red]")
        sys.exit(1)
    
    console.print(f"[blue]Test markdown file: {args.test_md}[/blue]")
    console.print(f"[blue]Output directory: {args.output_dir}[/blue]")
    
    # Download articles
    try:
        stats = download_test_articles(args.test_md, args.output_dir)
        
        # Create manifest
        if stats["successful"] > 0:
            create_download_manifest(stats, args.output_dir)
        
        # Exit with error code if all downloads failed
        if stats["successful"] == 0 and stats["total"] > 0:
            console.print("[red]All downloads failed![/red]")
            sys.exit(1)
        
        console.print(f"[bold green]✅ Download process completed![/bold green]")
        
    except KeyboardInterrupt:
        console.print("[yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()