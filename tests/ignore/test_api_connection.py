#!/usr/bin/env python3
"""
Test script for CDC Text Corpora LLM API connection.

This script tests the connection to the default LLM provider specified in .env
to verify that your API key is correctly configured.

Usage:
    python tests/test_api_connection.py
    
Or with uv:
    uv run tests/test_api_connection.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.qa.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

console = Console()

def main():
    """Test API connection for the default LLM provider."""
    
    console.print(Panel(
        "[bold blue]CDC Text Corpora - LLM API Connection Test[/bold blue]",
        title="🧪 API Test",
        border_style="blue"
    ))
    
    # Get default provider and model from environment
    default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    
    console.print(f"Testing connection to: [cyan]{default_provider.upper()}[/cyan] - [magenta]{default_model}[/magenta]\n")
    
    try:
        # Create a minimal corpus manager (we don't need data for API testing)
        corpus = CDCCorpus()
        
        # Initialize RAG engine with default settings
        rag_engine = RAGEngine(corpus)
        
        console.print("🔄 Testing LLM connection...")
        
        # Test the connection
        result = rag_engine.test_llm_connection()
        
        # Display results
        if result['success']:
            console.print("[bold green]✅ Connection successful![/bold green]")
            console.print(f"[dim]Response time: {result['response_time']}s[/dim]")
            
            console.print(Panel(
                f"[dim]{result['response']}[/dim]",
                title="LLM Response",
                border_style="green"
            ))
            
            console.print("\n[bold green]🎉 Your API connection is working![/bold green]")
            console.print("[dim]You're ready to use the CDC Text Corpora RAG system![/dim]")
            
        else:
            console.print("[bold red]❌ Connection failed![/bold red]")
            
            if not result['api_key_configured']:
                env_var = f"{default_provider.upper()}_API_KEY"
                console.print(Panel(
                    f"[yellow]API key not configured for {default_provider.upper()}[/yellow]\n\n"
                    f"1. Get your API key:\n"
                    f"   • OpenAI: https://platform.openai.com/api-keys\n"
                    f"   • Anthropic: https://console.anthropic.com/\n\n"
                    f"2. Add it to your .env file:\n"
                    f"   [code]{env_var}=your_api_key_here[/code]\n\n"
                    f"3. Restart this test",
                    title="Setup Required",
                    border_style="yellow"
                ))
            else:
                console.print(Panel(
                    f"[red]Error: {result['error']}[/red]",
                    title="Connection Error",
                    border_style="red"
                ))
                
    except Exception as e:
        console.print(f"[bold red]❌ Test failed with error:[/bold red]")
        console.print(Panel(
            f"[red]{str(e)}[/red]",
            title="Error",
            border_style="red"
        ))

if __name__ == "__main__":
    main()