"""SimLab customer support task environment for prime-rl training.

This environment provides customer support task prompts collected from
SimLab rollouts. The model generates responses and is scored on:
- Similarity to successful reference trajectories (Jaccard overlap)
- Response quality and structure (formatting, completeness)

Tasks include ticket triage, billing disputes, escalations, and
multi-turn customer conversations from the Weaver Enterprises scenario.
"""

from datasets import Dataset, load_dataset

from verifiers.envs.environment import Environment
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric


# --- Embedded dataset of SimLab customer support prompts ---
# These are extracted from SimLab task bundles and successful rollouts.
# For larger datasets, replace with a HuggingFace dataset ID.
SIMLAB_TASKS = [
    {
        "question": (
            "You've received an urgent report from James Wilson at Wilson Retail Group "
            "about intermittent sync failures in their platform integration during peak "
            "sales, causing critical disruptions to order processing and inventory sync. "
            "This is a VIP enterprise account with a 2-hour SLA. First, email Amanda "
            "Reeves to confirm the current account status and any ongoing issues she's "
            "aware of. Then create or locate the helpdesk ticket and escalate to Marcus "
            "Chen via Chat with full business context, including details Amanda provides. "
            "Keep James informed of progress throughout."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-enterprise-escalation",
    },
    {
        "question": (
            "Review Karen Mitchell's billing dispute for Invoice #INV-2024-5847. Contact "
            "Diana Walsh to get the exact details of the billing error and confirm the "
            "correct amount. Then contact James Foster to get his approval for the "
            "corrected invoice amount and any customer compensation or credit he is "
            "willing to authorize. Once you have both the error details from Diana and "
            "the approved corrected amount and credits from James, send Karen an apology "
            "email that includes the specific explanation of the error, the corrected "
            "invoice, and details of any credit or compensation approved."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-billing-dispute",
    },
    {
        "question": (
            "Karen Mitchell reported an account access issue 18 hours ago and hasn't "
            "received a response — we're at risk of breaching the 24-hour SLA. She's "
            "also posted frustration in the support-escalations channel. Contact Sarah "
            "Johnson to get her recommendation on which support agent should handle this "
            "based on current workload and expertise. Then reach out to Marcus Chen to "
            "determine if the 'Invalid credentials' error requires backend investigation "
            "or if standard account recovery will resolve it. Once you have their "
            "guidance, assign the ticket to the appropriate agent and send Karen an "
            "acknowledgment email that addresses her frustration and outlines next steps."
        ),
        "answer": "",
        "info": {},
        "task": "simlab-sla-breach",
    },
]


def _extract_text(completion: object) -> str:
    """Extract all text from a verifiers completion.

    The completion can be a plain string or a list of message objects.
    For message lists, we concatenate all assistant content AND
    reasoning_content so the rubric can score thinking models too.
    """
    if isinstance(completion, str):
        return completion

    parts: list[str] = []
    if isinstance(completion, list):
        for msg in completion:
            # Handle both dict and message objects
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if content:
                parts.append(str(content))
            # Also grab reasoning/thinking content
            reasoning = getattr(msg, "reasoning_content", None) or (msg.get("reasoning_content") if isinstance(msg, dict) else None)
            if reasoning:
                parts.append(str(reasoning))

    return "\n".join(parts)


def _quality_reward(completion: object, **kwargs: object) -> float:
    """Reward for well-structured, substantive responses."""
    text = _extract_text(completion)
    if not text or not text.strip():
        return 0.0

    text = text.strip()
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


def _completeness_reward(completion: object, question: str, **kwargs: object) -> float:
    """Reward for addressing all parts of the task instruction."""
    text = _extract_text(completion)
    if not text or not question:
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
        if action.lower() in text.lower()
    )

    return addressed / len(required_actions)


def load_environment(
    dataset_name: str | None = None,
    dataset_split: str = "train",
    system_prompt: str | None = None,
    **kwargs,
) -> Environment:
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

    parser = Parser()

    rubric = Rubric(parser=parser)
    rubric.add_reward_func(_quality_reward, weight=0.5)
    rubric.add_reward_func(_completeness_reward, weight=0.5)

    env = SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
