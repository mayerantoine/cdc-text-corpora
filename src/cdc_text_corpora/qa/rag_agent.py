"""
Agentic RAG implementation for CDC Text Corpora.

This module implements a multi-agent system for research question answering
based on the CDC Text Corpora dataset. The system includes specialized agents
for search, evidence analysis, and answer generation.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from typing import TypedDict
import asyncio

from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.qa.rag_engine import RAGEngine
from agents import Agent, Runner, RunContextWrapper, function_tool
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box


class SessionStatus(TypedDict):
    """Status tracking for agent session."""
    Paper: int
    Relevant: int
    Evidence: int


@dataclass
class SessionState:
    """Session state management for agentic RAG workflow."""
    original_question: str
    updated_question: str
    search_results: Optional[List] = field(default_factory=list)
    evidence_library: List[Tuple] = field(default_factory=list)
    status: Optional[Dict[str, int]] = field(default_factory=lambda: {'Paper': 0, 'Relevant': 0, 'Evidence': 0})


class EvidenceSummary(BaseModel):
    """Structured output for evidence analysis."""
    relevant_information_summary: str = Field(description="Summary of the evidence or 'Not applicable'")
    score: int = Field(description="A score from 1-10 indicating relevance to question")


class AgentConfig:
    """Configuration for agentic RAG system."""
    
    def __init__(
        self,
        collection_filter: str = 'pcd',
        relevance_cutoff: int = 8,
        search_k: int = 10,
        max_evidence_pieces: int = 5,
        max_search_attempts: int = 3,
        model_name: str = "gpt-4o-mini"
    ):
        self.collection_filter = collection_filter
        self.relevance_cutoff = relevance_cutoff
        self.search_k = search_k
        self.max_evidence_pieces = max_evidence_pieces
        self.max_search_attempts = max_search_attempts
        self.model_name = model_name


class AgentToolFactory:
    """Factory class for creating agent tools."""
    
    def __init__(self, corpus: CDCCorpus, rag_engine: RAGEngine, config: AgentConfig):
        self.corpus = corpus
        self.rag_engine = rag_engine
        self.config = config
        self._evidence_agent = None
        self._console = Console()
    
    @property
    def evidence_agent(self):
        """Lazy initialization of evidence agent."""
        if self._evidence_agent is None:
            instructions = (
                "You are a helpful research librarian assistant. Your role is to summarize chunk of evidence from literature. "
                "Summarize the text below to help answer a question. Do not directly answer the question, "
                "instead summarize to give evidence to help answer the question. Reply 'Not applicable' if text is irrelevant. "
                "Use 2-3 sentences. At the end of your response, provide a score from 1-10 on a newline indicating relevance to question. "
                "Do not explain your score."
            )
            
            self._evidence_agent = Agent(
                name="EvidenceAgent",
                instructions=instructions,
                model=self.config.model_name,
                output_type=EvidenceSummary
            )
        return self._evidence_agent
    
    def create_search_tool(self):
        """Create the search tool for finding relevant papers."""
        
        @function_tool
        async def search(state: "RunContextWrapper[SessionState]", question: str) -> str:
            """Use this tool to search for papers content to help answer the question."""
            
            # Update session state
            if state.context.original_question == "":
                state.context.original_question = question
            else:
                state.context.updated_question = question

            self._console.print(f"🟢 [Search] Starting paper search for question: [blue]{question}[/blue]")
            
            # Check if vector index exists, if not exit and ask to create it

            
            # Perform semantic search
            results = self.rag_engine.semantic_search(
                query=question,
                k=self.config.search_k,
                collection_filter=self.config.collection_filter
            )

            count_results = len(results)
            state.context.search_results.extend(results)
            state.context.status['Paper'] += count_results

            self._console.print(f"🟢 [Search] Paper search returned [yellow]{count_results}[/yellow] passages from papers")
            self._print_status(state.context.status)
            
            return f"🟢 [Search] Found [yellow]{count_results}[/yellow] text passages from the papers that semantically matches and can help answer the question."
        
        return search
    
    def create_evidence_tool(self):
        """Create the evidence gathering tool."""
        
        async def evidence_summary(evidence: str, question: str) -> EvidenceSummary:
            """Use the evidence agent to gather relevance information about search results."""
            user_instructions = (
                f"Summarize the text below to help answer a question. "
                f"### Evidence: {evidence} #### "
                f"#### Question: {question} #### "
                f"Relevant Information Summary: "
            )
            
            result = await Runner.run(self.evidence_agent, input=user_instructions)
            return result.final_output

        @function_tool
        async def gather_evidence(state: "RunContextWrapper[SessionState]", question: str) -> str:
            """Use this tool to gather evidence to help answer the question."""

            self._console.print(f"🟢 [Gather] Gathering evidence for question: [blue]{question}[/blue]")
            chunks = state.context.search_results

            # Process evidence in parallel
            tasks = [
                asyncio.create_task(evidence_summary(item['title'] + item['content'], question)) 
                for item in chunks
            ]
            results = await asyncio.gather(*tasks)
            self._console.print(f"🟢 [Gather] Finished gathering evidence for question: [blue]{question}[/blue]")

            # Filter high-quality evidence
            top_evidence_context = [
                (result.score, result.relevant_information_summary) 
                for result in results 
                if result.score >= self.config.relevance_cutoff
            ]
            count_top_evidence = len(top_evidence_context)

            # Update session state
            state.context.evidence_library.extend(top_evidence_context)
            state.context.status['Evidence'] = len(state.context.evidence_library)
            state.context.status['Relevant'] = len(state.context.evidence_library)

            best_evidence = "\n".join([evidence[1] for evidence in state.context.evidence_library])
            self._print_status(state.context.status)
            
            return f"🟢 [Gather] Found and added [yellow]{count_top_evidence}[/yellow] pieces of evidence relevant to the question. Best evidences: {best_evidence}."
        
        return gather_evidence
    
    def create_answer_tool(self):
        """Create the answer generation tool."""
        
        def get_answer_instructions(state: RunContextWrapper[SessionState], agent) -> str:
            """Generate dynamic instructions for answer agent."""
            context_evidence = "\n".join([evidence[1] for evidence in state.context.evidence_library])

            instructions = (
                "Write an answer for the question below based on the provided context. "
                "If the context provides insufficient information, reply 'I cannot answer'. "
                "Answer in an unbiased, comprehensive, and scholarly tone. "
                "If the question is subjective, provide an opinionated answer in the concluding 1-2 sentences."
            )
            instructions += f"\n## Context: {context_evidence}"
            instructions += f"\n## Question: {state.context.original_question}"

            return instructions

        answer_agent = Agent[SessionState](
            name="AnswerAgent",
            instructions=get_answer_instructions,
            model=self.config.model_name,
        )

        generate_answer = answer_agent.as_tool(
            tool_name="generate_answer",
            tool_description="Use this tool to generate a proposed answer to the question when you have collected enough evidence"
        )
        
        return generate_answer
    
    def _print_status(self, status: Dict[str, int]) -> None:
        """Print current session status."""
        self._console.print(f"🟢 [Status] Paper Count=[yellow]{status.get('Paper')}[/yellow] | Relevant Papers=[yellow]{status.get('Relevant')}[/yellow] | Current Evidence=[yellow]{status.get('Evidence')}[/yellow]")


class AgenticRAG:
    """
    Agentic RAG system for CDC Text Corpora research.
    
    This class implements a multi-agent workflow for answering research questions
    by coordinating specialized agents for search, evidence analysis, and synthesis.
    """
    
    def __init__(
        self, 
        corpus: Optional[CDCCorpus] = None, 
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            corpus: CDCCorpus instance. If None, will create one.
            config: AgentConfig instance. If None, will use defaults.
        """
  
        self.corpus = corpus or CDCCorpus()
        self.rag_engine = RAGEngine(self.corpus)
        self.config = config or AgentConfig()
        
        # Initialize tool factory
        self.tool_factory = AgentToolFactory(self.corpus, self.rag_engine, self.config)
        
        # Create orchestrator agent
        self._orchestrator_agent = None
    
    def _get_collection_instructions(self) -> str:
        """Get collection-specific instructions."""
        collection_map = {
            'pcd': 'Preventing Chronic Disease',
            'eid': 'Emerging Infectious Diseases', 
            'mmwr': 'Morbidity and Mortality Weekly Report',
            'all': 'CDC'
        }
        
        collection_name = collection_map.get(self.config.collection_filter, 'CDC')
        
        return f"""
            You are a senior researcher AI assistant of the {collection_name} journal. Your role is to answer questions based on evidence in the journal papers. 
            Answer in a direct and concise tone. Your audience is an expert, so be highly specific. If there are ambiguous terms or acronyms, first define them.
            You have access to three tools: search, gather_evidence and generate_answer.
            Search for papers, gather evidence, and answer. If you do not have enough evidence,
            you can search for more papers (preferred) or gather more evidence with a different phrase. 
            You may rephrase or break-up the question in those steps. Once you have {self.config.max_evidence_pieces} or more pieces of evidence from multiple sources, 
            or you have tried more than {self.config.max_search_attempts} times, call generate_answer tool. You may reject the answer and try again if it is incomplete.
            Important: remember to answer if you cannot find enough evidence.
            """
    
    @property
    def orchestrator_agent(self):
        """Lazy initialization of orchestrator agent."""
        if self._orchestrator_agent is None:
            # Get all tools
            search_tool = self.tool_factory.create_search_tool()
            evidence_tool = self.tool_factory.create_evidence_tool()
            answer_tool = self.tool_factory.create_answer_tool()
            
            collection_name = self.config.collection_filter.upper() if self.config.collection_filter != 'all' else 'CDC'
            
            self._orchestrator_agent = Agent[SessionState](
                name=f"{collection_name}Agent",
                instructions=self._get_collection_instructions(),
                model=self.config.model_name,
                tools=[search_tool, evidence_tool, answer_tool]
            )
        
        return self._orchestrator_agent
    
    def create_session_state(self) -> SessionState:
        """Create a new session state."""
        return SessionState(
            original_question="",
            updated_question="",
            search_results=[],
            evidence_library=[],
            status={'Paper': 0, 'Relevant': 0, 'Evidence': 0}
        )
    
    async def ask_question(self, question: str, max_turns: int = 10) -> str:
        """
        Ask a research question using the agentic workflow.
        
        Args:
            question: The research question to answer
            max_turns: Maximum number of agent decision cycles
            
        Returns:
            The generated answer
        """
        # Initialize session state
        session_state = self.create_session_state()
        
        # Run the agentic workflow
        result = await Runner.run(
            self.orchestrator_agent,
            input=question,
            context=session_state,
            max_turns=max_turns
        )
        
        return result.final_output
    
    def get_session_status(self, session_state: SessionState) -> Dict[str, int]:
        """Get current session status."""
        return session_state.status.copy()
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check if required data is available for the agentic pipeline.
        
        Returns:
            Dictionary with data availability status
        """
        status = {
            "parsed_articles_available": False,
            "collections_found": [],
            "total_articles": 0,
            "vector_index_exists": False,
            "recommendations": []
        }
        
        # Check for parsed JSON files
        json_parsed_dir = self.corpus.get_data_directory() / "json-parsed"
        
        if json_parsed_dir.exists():
            # Determine collections to check based on config
            collections_to_check = (
                [self.config.collection_filter] 
                if self.config.collection_filter and self.config.collection_filter != 'all' 
                else ['pcd', 'eid', 'mmwr']
            )
            
            for collection in collections_to_check:
                # Look for any language files for this collection
                pattern = f"{collection}_*_*.json"
                json_files = list(json_parsed_dir.glob(pattern))
                if json_files:
                    status["collections_found"].append(collection)
            
            if status["collections_found"]:
                status["parsed_articles_available"] = True
                
                # Count total articles using the corpus iterable
                try:
                    articles = self.corpus.load_json_articles_as_iterable(
                        collection=self.config.collection_filter if self.config.collection_filter != 'all' else None,
                        language='en'  # Default to English for counting
                    )
                    status["total_articles"] = len(articles)
                except Exception:
                    status["total_articles"] = 0
        
        # Check for vector index
        try:
            vector_stats = self.rag_engine.get_vectorstore_stats()
            if "total_documents" in vector_stats and vector_stats["total_documents"] > 0:
                status["vector_index_exists"] = True
        except Exception:
            status["vector_index_exists"] = False
        
        # Generate recommendations
        if not status["parsed_articles_available"]:
            status["recommendations"].append("Parse some collections first using: cdc-corpus parse --collection <name>")
        
        if not status["vector_index_exists"] and status["parsed_articles_available"]:
            status["recommendations"].append("Articles will be indexed automatically when you start the Q&A session")
        
        return status
    
    def display_pipeline_status(self, console: Optional[Console] = None) -> Dict[str, Any]:
        """Display the current agentic pipeline status."""
        if console is None:
            console = Console()
            
        status = self.check_data_availability()
        
        # Use AgenticRAG with interactive wrapper
        console.print(Panel(
                "[bold green]🤖 Agentic RAG Mode[/bold green]\n"
                "[dim]Multi-agent system for advanced research question answering[/dim]",
                title="Agentic Mode",
                border_style="green"
            ))
        
        # Create status table
        table = Table(title="Agentic Pipeline Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # LLM Connection
        llm_status = "✅ Ready" if self.rag_engine else "❌ Not initialized"
        llm_details = f"{self.rag_engine.llm_provider} - {self.rag_engine.llm_model_name}" if self.rag_engine else ""
        table.add_row("LLM Connection", llm_status, llm_details)
        
        # Agent Configuration
        config_details = f"Max evidence: {self.config.max_evidence_pieces}, Cutoff: {self.config.relevance_cutoff}"
        table.add_row("Agent Config", "✅ Configured", config_details)
        
        # Parsed Articles
        articles_status = "✅ Available" if status["parsed_articles_available"] else "❌ Missing"
        articles_details = f"{len(status['collections_found'])} collections, {status['total_articles']} articles"
        if status["collections_found"]:
            articles_details += f" ({', '.join(status['collections_found']).upper()})"
        table.add_row("Parsed Articles", articles_status, articles_details)
        
        # Vector Index
        index_status = "✅ Ready" if status["vector_index_exists"] else "⏳ Will be created"
        index_details = "Existing index found" if status["vector_index_exists"] else "Auto-indexing on first use"
        table.add_row("Vector Index", index_status, index_details)
        
        # Collection Filter
        filter_details = f"Collection: {self.config.collection_filter.upper() if self.config.collection_filter != 'all' else 'ALL'}"
        filter_details += f", Model: {self.config.model_name}"
        table.add_row("Configuration", "ℹ️  Active", filter_details)
        
        console.print(table)
        
        # Show recommendations if any
        if status["recommendations"]:
            console.print("\n[yellow]📋 Recommendations:[/yellow]")
            for rec in status["recommendations"]:
                console.print(f"  • {rec}")
        
        return status
    
    def ensure_vector_index(self, console: Optional[Console] = None) -> bool:
        """
        Ensure that articles are indexed in the vector database.
        
        Args:
            console: Optional console for user interaction and output
            
        Returns:
            True if indexing is successful or already exists, False otherwise
        """
        if console is None:
            console = Console()
            
        # Check if index already exists
        try:
            stats = self.rag_engine.get_vectorstore_stats()
            if "total_documents" in stats and stats["total_documents"] > 0:
                console.print(f"[green]✅ Vector index already exists ({stats['total_documents']} documents)[/green]")
                return True
        except Exception:
            pass
        
        # Check if we have parsed articles to index
        status = self.check_data_availability()
        if not status["parsed_articles_available"]:
            console.print("[red]❌ No parsed articles found. Please parse collections first.[/red]")
            return False
        
        # Ask user if they want to index
        console.print(f"\n[yellow]📚 Found {status['total_articles']} articles to index for agentic search[/yellow]")
        should_index = Confirm.ask("Would you like to index these articles for semantic search?", default=True)
        
        if not should_index:
            console.print("[yellow]⚠️  Skipping indexing. Agentic search will have limited functionality.[/yellow]")
            return False
        
        # Perform indexing
        try:
            console.print("[yellow]🔄 Indexing articles for agentic search...[/yellow]")
            
            # Use the collection filter from config
            collection_filter = (
                self.config.collection_filter 
                if self.config.collection_filter and self.config.collection_filter != 'all'
                else None
            )
            
            index_stats = self.rag_engine.index_articles(
                collection=collection_filter,
                language='en'  # Default to English for agentic indexing
            )
            
            console.print(f"[green]✅ Successfully indexed {index_stats['articles_processed']} articles into {index_stats['total_chunks']} chunks[/green]")
            console.print("[green]🤖 Agentic search is now ready![/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Indexing failed: {e}[/red]")
            console.print("[yellow]⚠️  Continuing without vector index. Search functionality will be limited.[/yellow]")
            return False
    
    def run_interactive_loop(self, console: Optional[Console] = None) -> None:
        """Run interactive loop for agentic RAG mode."""
        if console is None:
            console = Console()
            
        console.print(Panel(
            "[bold green]🎯 Interactive Agentic Q&A Session[/bold green]\n"
            "[dim]Ask research questions. The agents will search, gather evidence, and provide comprehensive answers.\n"
            "Type 'quit', 'exit', or 'q' to stop.[/dim]",
            title="Agentic Q&A Mode", 
            border_style="green"
        ))
        
        question_count = 0
        
        while True:
            try:
                # Get question from user
                question = Prompt.ask(f"\n[bold cyan]Research Question #{question_count + 1}[/bold cyan]")
                
                # Check for exit commands
                if question.lower().strip() in ['quit', 'exit', 'q', '']:
                    break
                
                # Show processing indicator
                console.print("[yellow]🔄 Processing with multi-agent system...[/yellow]")
                
                # Run async agentic RAG
                try:
                    answer = asyncio.run(self.ask_question(question, max_turns=10))
                    
                    # Display answer
                    console.print(Panel(
                        f"[cyan]{answer}[/cyan]",
                        title="🤖 Agentic Answer",
                        border_style="magenta"
                    ))
                    
                    question_count += 1
                    
                except Exception as e:
                    console.print(f"[red]❌ Error processing question: {e}[/red]")
                    continue
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]❌ Error in interactive loop: {e}[/red]")
                continue
        
        console.print(f"\n[green]👋 Agentic Q&A session ended. Answered {question_count} questions.[/green]")
    
    def run(self, console: Optional[Console] = None) -> None:
        """Run the complete agentic RAG pipeline."""
        if console is None:
            console = Console()
            
        try:
            # Display initial status
            status = self.display_pipeline_status(console)
            
            # Check if we can proceed
            if not status["parsed_articles_available"]:
                console.print("\n[red]❌ Cannot start agentic Q&A session without parsed articles.[/red]")
                console.print("[yellow]Please run: cdc-corpus parse --collection <name> first[/yellow]")
                return
            
            # Test LLM connection
            console.print("\n[yellow]🔄 Testing LLM connection...[/yellow]")
            test_result = self.rag_engine.test_llm_connection()
            
            if not test_result['success']:
                console.print(f"[red]❌ LLM connection failed: {test_result['error']}[/red]")
                return
            
            console.print("[green]✅ LLM connection successful[/green]")
            
            # Ensure vector index exists
            if not self.ensure_vector_index(console):
                # User chose not to index or indexing failed
                # We can still proceed but with limited functionality
                pass
            
            # Start interactive Q&A
            self.run_interactive_loop(console)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Agentic pipeline interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Agentic pipeline error: {e}[/red]")
            raise


async def main():
    """Example usage of the AgenticRAG system."""
    # Configuration
    config = AgentConfig(
        collection_filter='pcd',
        relevance_cutoff=8,
        search_k=10,
        max_evidence_pieces=5,
        max_search_attempts=3
    )
    
    # Initialize agentic RAG
    agentic_rag = AgenticRAG(config=config)
    
    # Ask question
    question = "What are the most common methods used in diabetes prevention to support adolescents in rural areas in the US?"
    
    print(f"Question: {question}\n")
    answer = await agentic_rag.ask_question(question, max_turns=10)
    print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())