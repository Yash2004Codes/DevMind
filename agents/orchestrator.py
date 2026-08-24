"""
agents/orchestrator.py — LangGraph Orchestrator

The "brain" of the DevMind system.

Given a natural language query from Claude Desktop (via MCP), the orchestrator:
  1. Classifies the query intent (which agents to involve)
  2. Calls the relevant agents in parallel using LangGraph
  3. Synthesizes the multi-agent results into one coherent answer

Graph nodes:
  classify → [memory, code, issue, decision] → synthesize

This uses a conditional routing pattern — not all agents are called for
every query. The classifier routes to only the relevant subgraph.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agents.memory_agent   import MemoryAgent
from agents.code_agent     import CodeAgent
from agents.issue_agent    import IssueAgent
from agents.decision_agent import DecisionAgent


# ── LangGraph State ───────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    """Shared state that flows through the LangGraph nodes."""
    query:            str
    intent:           list[str]
    memory_results:   list[dict]
    code_results:     list[dict]
    issue_results:    list[dict]
    decision_results: list[dict]
    final_answer:     str


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    LangGraph-based multi-agent orchestrator.
    Uses Groq (free) with Llama 3.1 70B for routing and synthesis.
    """

    # Groq free tier models — llama-3.1-70b is the most capable free model
    MODEL = "llama-3.1-70b-versatile"

    def __init__(
        self,
        memory_agent:   MemoryAgent,
        code_agent:     CodeAgent,
        issue_agent:    IssueAgent,
        decision_agent: DecisionAgent,
    ):
        self.memory_agent   = memory_agent
        self.code_agent     = code_agent
        self.issue_agent    = issue_agent
        self.decision_agent = decision_agent

        api_key = os.environ.get("GROQ_API_KEY", "")

        self.router_llm = ChatGroq(
            model       = self.MODEL,
            api_key     = api_key,
            temperature = 0.1,
            max_tokens  = 512,
        )
        self.synth_llm = ChatGroq(
            model       = self.MODEL,
            api_key     = api_key,
            temperature = 0.3,
            max_tokens  = 2048,
        )

        self._graph = self._build_graph()


    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _classify_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Use the LLM to decide which agent domains are relevant for this query.
        Returns a list of domain names to query.
        """
        prompt = f"""You are a query router for a developer productivity system.
Given the user's query, decide which knowledge domains are relevant.

Domains:
- memory: past conversation history, things the user discussed before
- code: GitHub PRs, commits, file changes, code authorship
- issues: GitHub Issues, bug reports, feature requests
- decisions: architectural decisions, "why did we choose X" questions

Query: "{state['query']}"

Respond with ONLY a JSON array of relevant domain names from: ["memory", "code", "issues", "decisions"]
Examples:
- "who worked on the auth module" → ["code"]  
- "why did we switch to PostgreSQL" → ["decisions", "code", "memory"]
- "find bugs similar to this error" → ["issues"]
- "what did I tell you about the API design" → ["memory"]
- "summarize last month's changes" → ["code", "issues", "decisions"]
"""
        response = self.router_llm.invoke([HumanMessage(content=prompt)])
        try:
            intent = json.loads(response.content.strip())
            if not isinstance(intent, list):
                intent = ["memory", "code", "issues", "decisions"]
        except Exception:
            intent = ["memory", "code", "issues", "decisions"]

        return {**state, "intent": intent}

    def _memory_node(self, state: OrchestratorState) -> OrchestratorState:
        """Query the Memory Agent if 'memory' is in the intent."""
        if "memory" not in state.get("intent", []):
            return state
        results = self.memory_agent.search_memories(state["query"], top_k=3)
        return {**state, "memory_results": results}

    def _code_node(self, state: OrchestratorState) -> OrchestratorState:
        """Query the Code Agent if 'code' is in the intent."""
        if "code" not in state.get("intent", []):
            return state
        results = self.code_agent.search_code(state["query"], top_k=3)
        return {**state, "code_results": results}

    def _issue_node(self, state: OrchestratorState) -> OrchestratorState:
        """Query the Issue Agent if 'issues' is in the intent."""
        if "issues" not in state.get("intent", []):
            return state
        results = self.issue_agent.search_issues(state["query"], top_k=3)
        return {**state, "issue_results": results}

    def _decision_node(self, state: OrchestratorState) -> OrchestratorState:
        """Query the Decision Agent if 'decisions' is in the intent."""
        if "decisions" not in state.get("intent", []):
            return state
        results = self.decision_agent.search_decisions(state["query"], top_k=3)
        return {**state, "decision_results": results}

    def _synthesize_node(self, state: OrchestratorState) -> OrchestratorState:
        """
        Synthesize results from all queried agents into one final answer.
        This is the "reduce" step of the map-reduce pattern.
        """
        context_parts = []

        if state.get("memory_results"):
            context_parts.append(
                "📝 PAST CONVERSATIONS:\n"
                + "\n".join(
                    f"  [{r['similarity']:.2f}] {r['user_prompt'][:100]}"
                    for r in state["memory_results"]
                )
            )

        if state.get("code_results"):
            context_parts.append(
                "💻 CODE / PRs / COMMITS:\n"
                + "\n".join(
                    f"  [{r['similarity']:.2f}] {r.get('chunk_type','').upper()} "
                    f"| {r.get('file_path','')} | by {r.get('author','?')} "
                    + (f"| PR #{r['pr_number']}" if r.get('pr_number') else "")
                    + f"\n    {r.get('preview','')[:150]}"
                    for r in state["code_results"]
                )
            )

        if state.get("issue_results"):
            context_parts.append(
                "🐛 GITHUB ISSUES:\n"
                + "\n".join(
                    f"  [{r['similarity']:.2f}] #{r.get('issue_number')} "
                    f"[{r.get('state','?')}] {r.get('title','')} "
                    f"| labels: {r.get('labels','')}"
                    for r in state["issue_results"]
                )
            )

        if state.get("decision_results"):
            context_parts.append(
                "🏛️ ARCHITECTURAL DECISIONS:\n"
                + "\n".join(
                    f"  [{r['similarity']:.2f}] {r.get('title','')}\n"
                    f"    Rationale: {r.get('rationale','')[:200]}"
                    for r in state["decision_results"]
                )
            )

        if not context_parts:
            final = (
                "I searched across all knowledge domains but found no relevant "
                "information. This might be a topic not yet indexed. Try running "
                "the GitHub ingester to index your repository first."
            )
            return {**state, "final_answer": final}

        context = "\n\n".join(context_parts)
        domains_used = ", ".join(state.get("intent", []))

        synth_prompt = f"""You are DevMind, an AI assistant with deep knowledge of a 
software project's history. You have just queried {domains_used} knowledge domains.

USER QUESTION: {state['query']}

RETRIEVED CONTEXT:
{context}

Based on the context above, provide a clear, helpful, and specific answer.
- Cite sources (PR numbers, issue numbers, authors) when relevant
- Be direct — don't pad with filler
- If the context is partially relevant, use what applies and note the gaps
- Format with markdown for readability"""

        response = self.synth_llm.invoke([HumanMessage(content=synth_prompt)])
        return {**state, "final_answer": str(response.content)}

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> any:
        """Build and compile the LangGraph state machine."""
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("classify",  self._classify_node)
        graph.add_node("memory",    self._memory_node)
        graph.add_node("code",      self._code_node)
        graph.add_node("issues",    self._issue_node)
        graph.add_node("decisions", self._decision_node)
        graph.add_node("synthesize",self._synthesize_node)

        # Set entry point
        graph.set_entry_point("classify")

        # All domain nodes run after classify (parallel fan-out)
        graph.add_edge("classify",  "memory")
        graph.add_edge("classify",  "code")
        graph.add_edge("classify",  "issues")
        graph.add_edge("classify",  "decisions")

        # All domain nodes feed into synthesize (fan-in)
        graph.add_edge("memory",    "synthesize")
        graph.add_edge("code",      "synthesize")
        graph.add_edge("issues",    "synthesize")
        graph.add_edge("decisions", "synthesize")

        graph.add_edge("synthesize", END)

        return graph.compile()

    # ── Public interface ──────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """
        Main entry point. Takes a natural language question, routes it
        through the multi-agent graph, returns the synthesized answer.

        Parameters
        ----------
        question : The user's natural language query

        Returns
        -------
        str — synthesized answer from all relevant agents
        """
        initial_state: OrchestratorState = {
            "query":            question,
            "intent":           [],
            "memory_results":   [],
            "code_results":     [],
            "issue_results":    [],
            "decision_results": [],
            "final_answer":     "",
        }
        final_state = self._graph.invoke(initial_state)
        return final_state["final_answer"]
