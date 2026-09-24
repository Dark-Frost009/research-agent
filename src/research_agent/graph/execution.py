"""Shared execution settings for the CLI and local web interface."""
from langchain_core.runnables import RunnableConfig

from research_agent.graph.budget import BudgetLimits


def research_run_config(limits: BudgetLimits) -> RunnableConfig:
    """Allow every budgeted iteration and finalization to finish.

    The sequential graph uses fewer than 20 steps per iteration; the extra
    headroom covers setup and finalization. This is an execution safety limit,
    not a replacement for the graph's provider-call and research budgets.
    Revisit it if the graph topology gains additional steps or inner loops.
    """
    return {"recursion_limit": max(100, limits.max_research_iterations * 20 + 10)}
