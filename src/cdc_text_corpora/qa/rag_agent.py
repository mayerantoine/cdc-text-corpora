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
import os

from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.qa.rag_engine import RAGEngine
from agents import Agent, Runner, RunContextWrapper, function_tool
from pydantic import BaseModel, Field
from rich.console import Console


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
        model_name: Optional[str] = None
    ):
        self.collection_filter = collection_filter
        self.relevance_cutoff = relevance_cutoff
        self.search_k = search_k
        self.max_evidence_pieces = max_evidence_pieces
        self.max_search_attempts = max_search_attempts
        
        # Use same environment variables as RAGEngine for consistency
        if model_name is None:
            # Get provider and model from environment (same as RAGEngine)
            provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai").lower()
            if provider == "openai":
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
            elif provider == "anthropic":
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "claude-3-5-sonnet")
            else:
                # Fallback to environment variable or default
                self.model_name = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        else:
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