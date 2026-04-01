"""NPC config generation for Sotopia-powered NPCs bundled with RocketChat."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_npc_profiles() -> list[dict[str, Any]]:
    """Load NPC profiles from the bundled catalog."""
    profiles_file = resources.files("simlab.catalog.npcs") / "profiles.json"
    return json.loads(profiles_file.read_text(encoding="utf-8"))


def generate_npc_credentials(profiles: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Generate ROCKETCHAT_NPC_CONFIGS JSON mapping profile_id to credentials.

    The seed container uses this to create RC users; the Sotopia runtime uses it
    to log in as those users.
    """
    configs: dict[str, dict[str, str]] = {}
    for p in profiles:
        profile_id = p["profile_id"]
        username = p.get("rocketchat_username", profile_id.replace("_", "."))
        configs[profile_id] = {
            "username": username,
            "password": "npc_pass_123",
            "name": f"{p['first_name']} {p['last_name']}",
            "email": p.get("email", f"{username}@weaverenterprises.com"),
        }
    return configs


def generate_npc_interaction_configs(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate SOTOPIA_NPC_INTERACTION_CONFIGS JSON array.

    Each entry bundles an agent profile, NPC profile, and environment profile
    matching the NPCInteractionConfig dataclass from the workspace controller.
    """
    agent_profile = {
        "pk": "agent_1",
        "first_name": "Simlab",
        "last_name": "Agent",
        "email": "agent@example.com",
        "age": 0,
        "occupation": "AI assistant",
        "gender": "",
        "gender_pronoun": "",
        "public_info": "",
        "big_five": "",
        "moral_values": [],
        "schwartz_personal_values": [],
        "personality_and_values": (
            "You are an AI agent being evaluated in a workplace chat simulation."
        ),
        "decision_making_style": "",
        "secret": "",
        "model_id": "",
        "mbti": "",
        "tag": "simlab",
    }

    configs: list[dict[str, Any]] = []
    for p in profiles:
        profile_id = p["profile_id"]
        username = p.get("rocketchat_username", profile_id.replace("_", "."))

        npc_profile = {
            "pk": profile_id,
            "first_name": p.get("first_name", ""),
            "last_name": p.get("last_name", ""),
            "email": p.get("email", f"{username}@weaverenterprises.com"),
            "age": p.get("age", 0),
            "occupation": p.get("occupation", ""),
            "gender": p.get("gender", ""),
            "gender_pronoun": p.get("gender_pronoun", ""),
            "public_info": p.get("public_info", ""),
            "big_five": p.get("big_five", ""),
            "moral_values": p.get("moral_values", []),
            "schwartz_personal_values": p.get("schwartz_personal_values", []),
            "personality_and_values": p.get("personality_and_values", ""),
            "decision_making_style": p.get("decision_making_style", ""),
            "secret": p.get("secret", ""),
            "model_id": p.get("model_id", ""),
            "mbti": p.get("mbti", ""),
            "tag": p.get("tag", "simlab"),
        }

        env_profile = {
            "pk": f"default-{profile_id}",
            "codename": f"default-{profile_id}",
            "source": "simlab",
            "scenario": "Workplace at Weaver Enterprises",
            "agent_goals": [
                "Interact with colleagues over workplace chat.",
                "You want to collaborate with the agent over workplace chat.",
            ],
            "relationship": 2,  # acquaintance
            "age_constraint": None,
            "occupation_constraint": None,
            "agent_constraint": [["agent_1"], [profile_id]],
            "tag": "simlab",
        }

        configs.append(
            {
                "agent_profile": agent_profile,
                "npc_profile": npc_profile,
                "env_profile": env_profile,
                "npc_profile_id": profile_id,
                "group_channel_names": [],
            }
        )

    return configs


NPC_CREDENTIALS_FILENAME = "rocketchat-npc-configs.json"
NPC_INTERACTION_CONFIGS_FILENAME = "sotopia-npc-interaction-configs.json"
_NPC_CONFIG_MOUNT_DIR = "/config/npc"


def build_npc_config_json(profiles: list[dict[str, Any]]) -> tuple[str, str]:
    """Build the NPC credentials and interaction configs JSON strings.

    Returns (npc_credentials_json, interaction_configs_json).
    """
    npc_credentials_json = json.dumps(generate_npc_credentials(profiles), indent=2)
    interaction_configs_json = json.dumps(generate_npc_interaction_configs(profiles), indent=2)
    return npc_credentials_json, interaction_configs_json


def inject_npc_env_vars(services: dict[str, dict[str, Any]], output_dir: str) -> tuple[str, str]:
    """Mount NPC config files into rocketchat-seed and sotopia-runtime services.

    Instead of passing large JSON payloads as environment variables (which can
    exceed the OS ARG_MAX limit), the configs are written to files that are
    bind-mounted into the containers.

    Returns (npc_credentials_json, interaction_configs_json) so the caller can
    write the files to the output directory.
    """
    profiles = load_npc_profiles()
    npc_credentials_json, interaction_configs_json = build_npc_config_json(profiles)

    creds_host_path = f"{output_dir}/{NPC_CREDENTIALS_FILENAME}"
    interaction_host_path = f"{output_dir}/{NPC_INTERACTION_CONFIGS_FILENAME}"

    creds_container_path = f"{_NPC_CONFIG_MOUNT_DIR}/{NPC_CREDENTIALS_FILENAME}"
    interaction_container_path = f"{_NPC_CONFIG_MOUNT_DIR}/{NPC_INTERACTION_CONFIGS_FILENAME}"

    if "rocketchat-seed" in services:
        env = services["rocketchat-seed"].setdefault("environment", {})
        env["ROCKETCHAT_NPC_CONFIGS_FILE"] = creds_container_path
        vols = services["rocketchat-seed"].setdefault("volumes", [])
        vols.append(f"{creds_host_path}:{creds_container_path}:ro")

    if "sotopia-runtime" in services:
        env = services["sotopia-runtime"].setdefault("environment", {})
        env["ROCKETCHAT_NPC_CONFIGS_FILE"] = creds_container_path
        env["SOTOPIA_NPC_INTERACTION_CONFIGS_FILE"] = interaction_container_path
        vols = services["sotopia-runtime"].setdefault("volumes", [])
        vols.append(f"{creds_host_path}:{creds_container_path}:ro")
        vols.append(f"{interaction_host_path}:{interaction_container_path}:ro")

    return npc_credentials_json, interaction_configs_json
