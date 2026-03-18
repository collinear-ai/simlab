"""Built-in preset configs for `simlab tasks-gen init`.

Each preset is a plain dict containing only the sections that simlab sends to the
server (toolset, agent, scenario, workspace, workflows, npcs, categories,
generation, pipeline).  Server-only fields (seed_data, evaluation, output) are
excluded.
"""

from __future__ import annotations

from typing import TypedDict


class AgentConfig(TypedDict):
    """Agent role and description."""

    role: str
    description: str


class ToolDefConfig(TypedDict):
    """Tool definition with name, description, and supported operations."""

    name: str
    description: str
    operations: list[str]


class ScenarioConfig(TypedDict):
    """Scenario context: name, role label, conventions, and policies."""

    name: str
    role_label: str
    conventions: str
    policies: list[str]


class WorkspaceConfig(TypedDict):
    """Workspace email settings."""

    email_domain: str
    agent_email: str


class WorkflowConfig(TypedDict):
    """Named workflow with ordered steps."""

    name: str
    steps: list[str]


class NpcConfig(TypedDict):
    """NPC role and typical asks."""

    role: str
    typical_asks: str


class CategoryConfig(TypedDict):
    """Task category with id and label."""

    id: str
    label: str


class GenerationConfig(TypedDict):
    """Task generation settings: count and complexity distribution."""

    num_tasks: int
    complexity: dict[str, float]


class PipelineConfig(TypedDict):
    """Pipeline model configuration."""

    model: str


class PresetConfig(TypedDict):
    """Full preset configuration sent to the server."""

    agent: AgentConfig
    toolset: list[ToolDefConfig]
    scenario: ScenarioConfig
    workspace: WorkspaceConfig
    workflows: list[WorkflowConfig]
    npcs: list[NpcConfig]
    categories: list[CategoryConfig]
    generation: GenerationConfig
    pipeline: PipelineConfig


RECRUITING: PresetConfig = {
    "agent": {
        "role": "HR recruiting coordinator",
        "description": (
            "Handles end-to-end recruiting workflows: scheduling interviews, "
            "managing candidate pipelines, coordinating offer discussions, "
            "and communicating with hiring managers and candidates."
        ),
    },
    "toolset": [
        {
            "name": "HRIS",
            "description": "Query/update employee records, job requisitions, candidate profiles",
            "operations": ["search", "read", "create", "update"],
        },
        {
            "name": "Email",
            "description": "Send and read emails",
            "operations": ["send", "read"],
        },
        {
            "name": "Calendar",
            "description": "View and manage calendar events",
            "operations": ["list", "create", "update", "delete"],
        },
        {
            "name": "Chat",
            "description": "Send messages in Rocket.Chat channels and DMs",
            "operations": ["send", "read"],
        },
    ],
    "scenario": {
        "name": "recruiting",
        "role_label": "HR recruiting professional",
        "conventions": (
            "- Always check all participants' calendars before scheduling\n"
            "- Never share compensation details in group channels\n"
            "- Document all candidate interactions in HRIS\n"
            "- Get manager approval before extending offers\n"
        ),
        "policies": [
            "Interviews must include at least one diverse panel member",
            "Offers require VP approval for >$200k total comp",
            "Candidate data must not be shared outside recruiting team",
        ],
    },
    "workspace": {
        "email_domain": "weaver.com",
        "agent_email": "hr@weaver.com",
    },
    "workflows": [
        {
            "name": "Schedule panel interview",
            "steps": [
                "Check interviewer availability on calendar",
                "Create calendar event with all panelists",
                "Send confirmation email to candidate",
                "Update candidate status in HRIS",
            ],
        },
        {
            "name": "Process offer rejection",
            "steps": [
                "Read candidate's rejection email",
                "Update candidate status to 'Declined' in HRIS",
                "Notify hiring manager via chat",
                "Archive requisition if no backup candidates",
            ],
        },
    ],
    "npcs": [
        {
            "role": "Hiring Manager",
            "typical_asks": "Scheduling preferences, candidate feedback, offer approvals",
        },
        {
            "role": "Candidate",
            "typical_asks": "Interview logistics, offer details, timeline questions",
        },
        {
            "role": "Recruiter Lead",
            "typical_asks": "Pipeline updates, escalations, compliance checks",
        },
    ],
    "categories": [
        {"id": "job_requisition", "label": "Job requisition"},
        {"id": "shortlist_resumes", "label": "Shortlist the resumes"},
        {"id": "schedule_interviews", "label": "Schedule the interviews"},
        {"id": "consolidate_feedback", "label": "Consolidate feedback"},
        {"id": "send_offers_or_rejections", "label": "Send out offers or rejections"},
        {"id": "negotiate_and_final_offer", "label": "Negotiate and send final offer"},
        {"id": "handle_candidate_rejections", "label": "Handle candidate rejections"},
        {"id": "candidate_questions", "label": "Handle questions from candidates"},
    ],
    "generation": {
        "num_tasks": 10,
        "complexity": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    },
    "pipeline": {
        "model": "claude-haiku-4-5",
    },
}


PEOPLE_MGMT: PresetConfig = {
    "agent": {
        "role": "HR people management coordinator",
        "description": (
            "Handles employee lifecycle operations: onboarding, performance management, "
            "compensation and benefits, employee relations, workforce planning, "
            "compliance, and offboarding."
        ),
    },
    "toolset": [
        {
            "name": "HRIS",
            "description": "Query/update employee records, enrollment records, HR policies",
            "operations": ["search", "read", "create", "update"],
        },
        {
            "name": "Email",
            "description": "Send and read emails",
            "operations": ["send", "read"],
        },
        {
            "name": "Calendar",
            "description": "View and manage calendar events",
            "operations": ["list", "create", "update", "delete"],
        },
        {
            "name": "Chat",
            "description": "Send messages in Rocket.Chat channels and DMs",
            "operations": ["send", "read"],
        },
    ],
    "scenario": {
        "name": "people_mgmt",
        "role_label": "HR people management professional",
        "conventions": (
            "- Always verify employee data in HRIS before taking action\n"
            "- Never share compensation details in group channels\n"
            "- Document all employee interactions in HRIS\n"
            "- Get manager approval for compensation changes\n"
            "- Follow escalation paths for disciplinary actions\n"
        ),
        "policies": [
            "PIPs require HR review and manager sign-off",
            "Compensation adjustments over 15% require VP approval",
            "Offboarding must include IT access revocation checklist",
            "Medical accommodations require documented approval chain",
        ],
    },
    "workspace": {
        "email_domain": "weaver.com",
        "agent_email": "hr@weaver.com",
    },
    "workflows": [
        {
            "name": "New employee onboarding",
            "steps": [
                "Create employee record in HRIS",
                "Schedule orientation meetings on calendar",
                "Send welcome email with first-week agenda",
                "Notify team via chat channel",
            ],
        },
        {
            "name": "Process compensation adjustment",
            "steps": [
                "Review current compensation in HRIS",
                "Get manager approval via chat",
                "Update compensation in HRIS",
                "Send confirmation email to employee",
            ],
        },
    ],
    "npcs": [
        {
            "role": "Direct Manager",
            "typical_asks": "Performance feedback, team updates, approval requests",
        },
        {
            "role": "Employee",
            "typical_asks": "Benefits questions, policy clarifications, accommodations",
        },
        {
            "role": "HR Business Partner",
            "typical_asks": "Compliance checks, escalations, workforce planning",
        },
    ],
    "categories": [
        {"id": "employee_onboarding", "label": "Employee onboarding"},
        {"id": "learning_and_development", "label": "Learning & development"},
        {"id": "performance_management", "label": "Performance management"},
        {"id": "compensation_and_benefits", "label": "Compensation & benefits"},
        {"id": "employee_relations", "label": "Employee relations"},
        {"id": "workforce_planning", "label": "Workforce planning"},
        {"id": "hr_business_partnering", "label": "HR business partnering"},
        {"id": "diversity_equity_and_inclusion", "label": "Diversity, equity & inclusion"},
        {
            "id": "employee_engagement_and_wellbeing",
            "label": "Employee engagement & wellbeing",
        },
        {"id": "offboarding", "label": "Offboarding"},
        {"id": "hr_compliance_and_policy", "label": "HR compliance & policy"},
    ],
    "generation": {
        "num_tasks": 10,
        "complexity": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    },
    "pipeline": {
        "model": "claude-haiku-4-5",
    },
}


CODING: PresetConfig = {
    "agent": {
        "role": "Software engineering assistant",
        "description": (
            "Completes end-to-end programming tasks in a sandboxed Linux workspace: "
            "reads requirements, clarifies ambiguities with stakeholders via Rocket.Chat, "
            "implements solutions across multiple files, runs tests and linters via bash, "
            "builds CLI tools / REST APIs / data-processing pipelines, and verifies web "
            "endpoints or UI changes in a headless browser."
        ),
    },
    "toolset": [
        {
            "name": "Coding",
            "description": (
                "Sandboxed Linux workspace (OpenHands) with filesystem and bash execution"
            ),
            "operations": ["run_bash", "read_file", "write_file", "list_files"],
        },
        {
            "name": "Browser",
            "description": (
                "Playwright-based headless Chromium for verifying web UIs and REST APIs"
            ),
            "operations": [
                "navigate",
                "click",
                "get_text",
                "fill",
                "screenshot",
                "scroll",
                "evaluate",
            ],
        },
        {
            "name": "Chat",
            "description": "Rocket.Chat for stakeholder coordination and NPC interaction",
            "operations": ["send", "read"],
        },
    ],
    "scenario": {
        "name": "coding",
        "role_label": "Software engineering professional",
        "conventions": (
            "- Read existing code and understand the codebase before making changes\n"
            "- Clarify ambiguous requirements with stakeholders on Rocket.Chat\n"
            "- Write tests for new functionality and run them before declaring done\n"
            "- Run linters and formatters to maintain code quality\n"
            "- Use proper error handling\n"
            "- Keep solutions focused — implement what's asked, don't over-engineer\n"
        ),
        "policies": [
            "All new functionality must have passing tests before completion",
            "REST APIs must include proper error responses and input validation",
            "CLI tools must support --help and exit with proper status codes",
            "Security-sensitive operations require input sanitization",
        ],
    },
    "workspace": {
        "email_domain": "",
        "agent_email": "",
    },
    "workflows": [
        {
            "name": "Build CLI application",
            "steps": [
                "Read task requirements and identify supported commands",
                "Create project structure",
                "Implement argument parsing",
                "Implement each command with proper error handling",
                "Write tests for each command and edge cases",
                "Run tests via bash to confirm all pass",
            ],
        },
        {
            "name": "Build REST API server",
            "steps": [
                "Read API specification and identify endpoints",
                "Set up Flask/FastAPI project with proper structure",
                "Define database models",
                "Implement CRUD endpoints with input validation",
                "Write integration tests for each endpoint",
                "Start server and verify endpoints in browser",
            ],
        },
    ],
    "npcs": [
        {
            "role": "Tech Lead",
            "typical_asks": (
                "Requirements clarification, architecture decisions, code review feedback"
            ),
        },
        {
            "role": "QA Engineer",
            "typical_asks": (
                "Test coverage requirements, bug reproduction details, regression checks"
            ),
        },
        {
            "role": "Product Manager",
            "typical_asks": (
                "Feature priorities, acceptance criteria clarification, deadline constraints"
            ),
        },
    ],
    "categories": [
        {"id": "cli_application", "label": "Build CLI applications"},
        {"id": "rest_api", "label": "Build REST API servers"},
        {"id": "data_processing", "label": "Parse, analyze, and transform data"},
        {"id": "bug_fix", "label": "Debug and fix reported bugs"},
        {"id": "feature_implementation", "label": "Implement features in existing codebases"},
        {"id": "web_verification", "label": "Build web services and verify via browser"},
        {"id": "refactoring", "label": "Refactor code for clarity or performance"},
    ],
    "generation": {
        "num_tasks": 10,
        "complexity": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    },
    "pipeline": {
        "model": "claude-haiku-4-5",
    },
}


CUSTOMER_SUPPORT: PresetConfig = {
    "agent": {
        "role": "Customer support agent",
        "description": (
            "Handles end-to-end customer support workflows: triaging incoming tickets, "
            "resolving issues using the knowledge base and saved replies, managing multi-turn "
            "customer conversations, escalating technical issues to engineering, enforcing SLA "
            "compliance, and coordinating cross-channel resolution."
        ),
    },
    "toolset": [
        {
            "name": "Helpdesk",
            "description": (
                "Frappe-based helpdesk for ticket lifecycle management, knowledge base, "
                "and saved replies"
            ),
            "operations": [
                "create_ticket",
                "get_ticket",
                "update_ticket",
                "list_tickets",
                "search_tickets",
                "assign_ticket",
                "send_ticket_email",
                "search_kb",
                "get_kb_article",
                "list_saved_replies",
                "get_sla",
                "check_sla_status",
            ],
        },
        {
            "name": "Chat",
            "description": (
                "Rocket.Chat for real-time customer conversations and internal coordination"
            ),
            "operations": ["send", "read"],
        },
        {
            "name": "Email",
            "description": "Send and read customer-facing and internal emails",
            "operations": ["send", "read"],
        },
    ],
    "scenario": {
        "name": "customer_support",
        "role_label": "Customer support professional",
        "conventions": (
            "- Always search the knowledge base before escalating to engineering\n"
            "- Use saved replies for greeting and acknowledgment\n"
            "- Follow SLA response-time requirements (Critical <1h, High <4h, Medium <8h)\n"
            "- Document all customer interactions as ticket comments\n"
            "- Escalate technical issues to engineering with full reproduction context\n"
            "- Use empathetic, professional language\n"
            "- VIP/Enterprise customers get priority queue\n"
            "- Never share internal escalation notes with customers\n"
        ),
        "policies": [
            "Critical-priority tickets require first response within 1 hour",
            "High-priority tickets require first response within 4 hours",
            "Billing adjustments over $500 require support manager approval",
            "VIP and Enterprise customers have priority queue access and dedicated SLA",
        ],
    },
    "workspace": {
        "email_domain": "weaverenterprises.com",
        "agent_email": "support@weaverenterprises.com",
    },
    "workflows": [
        {
            "name": "Triage and resolve incoming ticket",
            "steps": [
                "Read ticket details and identify customer tier",
                "Check SLA status and priority",
                "Search knowledge base for known solution",
                "Resolve with KB article or custom response",
                "Update ticket status to Resolved",
                "Send resolution email to customer",
            ],
        },
        {
            "name": "Escalate to engineering",
            "steps": [
                "Confirm issue is not in knowledge base",
                "Gather reproduction steps from customer conversation",
                "Send internal message tagging engineering lead",
                "Update ticket status to Escalated",
                "Inform customer of escalation with expected timeline",
            ],
        },
    ],
    "npcs": [
        {
            "role": "Regular Customer",
            "typical_asks": "Issue resolution, feature questions, billing inquiries",
        },
        {
            "role": "VIP/Enterprise Customer",
            "typical_asks": "Urgent production issues, dedicated SLA enforcement",
        },
        {
            "role": "Support Manager",
            "typical_asks": "Exception approvals, process guidance, escalation decisions",
        },
        {
            "role": "Engineering Lead",
            "typical_asks": "Bug triage, technical escalation review",
        },
    ],
    "categories": [
        {"id": "ticket_triage", "label": "Triage and route incoming tickets"},
        {"id": "ticket_resolution", "label": "Resolve tickets using KB"},
        {"id": "billing_inquiry", "label": "Handle billing questions and refund requests"},
        {"id": "technical_escalation", "label": "Escalate technical issues to engineering"},
        {"id": "manager_escalation", "label": "Route exception requests to support manager"},
        {"id": "multi_turn_conversation", "label": "Manage multi-turn customer conversations"},
        {"id": "vip_enterprise_support", "label": "Handle VIP and enterprise priority issues"},
        {"id": "sla_compliance", "label": "Monitor and enforce SLA compliance"},
    ],
    "generation": {
        "num_tasks": 10,
        "complexity": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    },
    "pipeline": {
        "model": "claude-haiku-4-5",
    },
}


HR: PresetConfig = {
    "agent": {
        "role": "HR coordinator",
        "description": (
            "Handles end-to-end HR workflows spanning recruiting and people management: "
            "scheduling interviews, managing candidate pipelines, coordinating offers, "
            "onboarding, performance management, compensation, employee relations, "
            "and communicating with hiring managers, candidates, and employees."
        ),
    },
    "toolset": [
        {
            "name": "HRIS",
            "description": (
                "Query/update employee records, job requisitions, candidate profiles, "
                "enrollment records, HR policies"
            ),
            "operations": ["search", "read", "create", "update"],
        },
        {
            "name": "Email",
            "description": "Send and read emails",
            "operations": ["send", "read"],
        },
        {
            "name": "Calendar",
            "description": "View and manage calendar events",
            "operations": ["list", "create", "update", "delete"],
        },
        {
            "name": "Chat",
            "description": "Send messages in Rocket.Chat channels and DMs",
            "operations": ["send", "read"],
        },
    ],
    "scenario": {
        "name": "hr",
        "role_label": "HR professional",
        "conventions": (
            "- Always verify employee/candidate data in HRIS before taking action\n"
            "- Check all participants' calendars before scheduling\n"
            "- Never share compensation details in group channels\n"
            "- Document all interactions in HRIS\n"
            "- Get manager approval before extending offers or adjusting compensation\n"
            "- Follow escalation paths for disciplinary actions\n"
        ),
        "policies": [
            "Interviews must include at least one diverse panel member",
            "Offers require VP approval for >$200k total comp",
            "Candidate data must not be shared outside recruiting team",
            "PIPs require HR review and manager sign-off",
            "Compensation adjustments over 15% require VP approval",
            "Offboarding must include IT access revocation checklist",
        ],
    },
    "workspace": {
        "email_domain": "weaverenterprises.com",
        "agent_email": "hr@weaverenterprises.com",
    },
    "workflows": [
        {
            "name": "Schedule panel interview",
            "steps": [
                "Check interviewer availability on calendar",
                "Create calendar event with all panelists",
                "Send confirmation email to candidate",
                "Update candidate status in HRIS",
            ],
        },
        {
            "name": "Process offer",
            "steps": [
                "Get manager approval via chat",
                "Create offer letter from template",
                "Send offer email to candidate",
                "Update candidate status in HRIS",
            ],
        },
        {
            "name": "New employee onboarding",
            "steps": [
                "Create employee record in HRIS",
                "Schedule orientation meetings on calendar",
                "Send welcome email with first-week agenda",
                "Notify team via chat channel",
            ],
        },
        {
            "name": "Process compensation adjustment",
            "steps": [
                "Review current compensation in HRIS",
                "Get manager approval via chat",
                "Update compensation in HRIS",
                "Send confirmation email to employee",
            ],
        },
    ],
    "npcs": [
        {
            "role": "Hiring Manager",
            "typical_asks": "Scheduling preferences, candidate feedback, offer approvals",
        },
        {
            "role": "Candidate",
            "typical_asks": "Interview logistics, offer details, timeline questions",
        },
        {
            "role": "Direct Manager",
            "typical_asks": "Performance feedback, team updates, approval requests",
        },
        {
            "role": "Employee",
            "typical_asks": "Benefits questions, policy clarifications, accommodations",
        },
        {
            "role": "HR Business Partner",
            "typical_asks": "Compliance checks, escalations, workforce planning",
        },
    ],
    "categories": [
        {"id": "schedule_interviews", "label": "Schedule interviews"},
        {"id": "send_offers_or_rejections", "label": "Send offers or rejections"},
        {"id": "employee_onboarding", "label": "Employee onboarding"},
        {"id": "performance_management", "label": "Performance management"},
        {"id": "compensation_and_benefits", "label": "Compensation & benefits"},
        {"id": "employee_relations", "label": "Employee relations"},
        {"id": "offboarding", "label": "Offboarding"},
        {"id": "hr_compliance_and_policy", "label": "HR compliance & policy"},
    ],
    "generation": {
        "num_tasks": 10,
        "complexity": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    },
    "pipeline": {
        "model": "claude-haiku-4-5",
    },
}


PRESETS: dict[str, PresetConfig] = {
    "hr": HR,
    "recruiting": RECRUITING,
    "people_mgmt": PEOPLE_MGMT,
    "coding": CODING,
    "customer_support": CUSTOMER_SUPPORT,
}
