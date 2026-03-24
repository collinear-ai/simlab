from __future__ import annotations

from simlab.composer.npc_config import generate_npc_credentials
from simlab.composer.npc_config import load_npc_profiles


def test_shared_npc_catalog_includes_crm_sales_manager() -> None:
    credentials = generate_npc_credentials(load_npc_profiles())

    assert credentials["jordan_miles"] == {
        "username": "jordan_miles",
        "password": "npc_pass_123",
        "name": "Jordan Miles",
        "email": "jordan.miles@weaverenterprises.com",
    }
