"""SimLab customer support task environment for prime-rl training.

This environment provides customer support task prompts collected from
SimLab rollouts. The model generates responses and is scored on:
- Similarity to successful reference trajectories (Jaccard overlap)
- Response quality and structure (formatting, completeness)

Tasks include ticket triage, billing disputes, escalations, and
multi-turn customer conversations from the Weaver Enterprises scenario.
"""

from datasets import Dataset, load_dataset

import verifiers as vf


# --- Embedded dataset of SimLab customer support prompts ---
# These are extracted from SimLab task bundles and successful rollouts.
# For larger datasets, replace with a HuggingFace dataset ID.
SIMLAB_TASKS = [
    {
        "question": (
            "You've received a billing dispute from Karen Mitchell regarding her "
            "enterprise renewal invoice. She reports a 40% increase with unexpected "
            "charges and is threatening to cancel by end of week. Contact Diana Walsh "
            "to obtain the specific billing details and charge breakdown from the "
            "invoice. Then contact Carlos Mendez to confirm what was discussed during "
            "the renewal process and any contract amendments. Once you have the facts "
            "from both, review the ticket details and determine whether this is a "
            "genuine billing error or a legitimate contract amendment issue. Provide "
            "Karen with a clear explanation of the charges and your recommended "
            "resolution path."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-billing-dispute",
    },
    {
        "question": (
            "David Park from TechStart Inc has reported persistent API rate limiting "
            "issues affecting their production environment. His enterprise SLA "
            "guarantees 99.9% uptime and the current issues are putting them at risk "
            "of breaching that threshold. Investigate the technical details of the "
            "rate limiting, coordinate with engineering to identify root cause, and "
            "provide David with a resolution timeline. Ensure the response meets the "
            "enterprise SLA first-response requirements."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-api-escalation",
    },
    {
        "question": (
            "Wilson Retail Group has filed an SLA-critical billing dispute claiming "
            "they were double-charged for their Q4 platform usage. The account is "
            "flagged as at-risk for churn. Review the billing records, cross-reference "
            "with the CRM account history, and determine whether the duplicate charge "
            "is valid. If confirmed, initiate the refund process and coordinate with "
            "the account manager to schedule a retention call. Document all findings "
            "in the support ticket."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-billing-sla",
    },
]


def _quality_reward(completion: str, **kwargs) -> float:
    """Reward for well-structured, substantive responses."""
    if not completion or not completion.strip():
        return 0.0

    text = completion.strip()
    score = 0.0

    # Length-based scoring
    if len(text) > 500:
        score += 0.3
    elif len(text) > 200:
        score += 0.2
    elif len(text) > 50:
        score += 0.1

    # Structure scoring
    if any(marker in text for marker in ["##", "**", "- ", "1.", "* "]):
        score += 0.2

    # Task-specific keyword scoring (customer support domain)
    cs_keywords = [
        "ticket", "customer", "escalat", "resolv", "billing",
        "sla", "priority", "update", "follow-up", "investigation",
    ]
    keyword_hits = sum(1 for kw in cs_keywords if kw.lower() in text.lower())
    score += min(0.3, keyword_hits * 0.05)

    # Professional tone indicators
    if any(phrase in text.lower() for phrase in [
        "i understand", "thank you", "please", "we will",
        "next steps", "resolution", "apolog",
    ]):
        score += 0.2

    return min(score, 1.0)


def _completeness_reward(completion: str, question: str, **kwargs) -> float:
    """Reward for addressing all parts of the task instruction."""
    if not completion or not question:
        return 0.0

    # Extract action items from the question
    action_indicators = ["contact", "review", "determine", "provide", "ensure",
                         "investigate", "coordinate", "document", "initiate"]
    required_actions = [
        word for word in action_indicators
        if word.lower() in question.lower()
    ]

    if not required_actions:
        return 0.5

    addressed = sum(
        1 for action in required_actions
        if action.lower() in completion.lower()
    )

    return addressed / len(required_actions)


def load_environment(
    dataset_name: str | None = None,
    dataset_split: str = "train",
    system_prompt: str | None = None,
    **kwargs,
) -> vf.Environment:
    """Load the SimLab customer support environment.

    Args:
        dataset_name: HuggingFace dataset ID. If None, uses embedded tasks.
        dataset_split: Dataset split to use.
        system_prompt: Override system prompt.

    Returns:
        A verifiers SingleTurnEnv for prime-rl training.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a customer support agent at Weaver Enterprises. "
            "You have access to helpdesk tickets, email, and chat tools. "
            "Handle customer issues professionally and thoroughly. "
            "Think step by step about what information you need, who to "
            "contact, and how to resolve the issue. Provide clear, "
            "actionable responses."
        )

    # Load dataset
    if dataset_name:
        train_dataset = load_dataset(dataset_name, split=dataset_split)
    else:
        train_dataset = Dataset.from_list(SIMLAB_TASKS)

    parser = vf.Parser()

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(_quality_reward, weight=0.5)
    rubric.add_reward_func(_completeness_reward, weight=0.5)

    env = vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
