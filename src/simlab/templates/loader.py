"""Scenario preset loader — loads pre-defined tool combinations from YAML."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic import Field


class TemplateDefinition(BaseModel):
    """A pre-defined combination of tools for a scenario."""

    name: str
    display_name: str
    description: str
    tools: list[str] = Field(default_factory=list)


class TemplateLoader:
    """Loads scenario preset templates from YAML files."""

    def __init__(self) -> None:
        """Initialize an empty template loader."""
        self._templates: dict[str, TemplateDefinition] = {}

    def load_all(self) -> None:
        """Load all template YAMLs from the templates/scenarios package directory."""
        scenarios_pkg = resources.files("simlab.templates") / "scenarios"
        for item in scenarios_pkg.iterdir():
            if hasattr(item, "name") and item.name.endswith(".yaml"):
                text = item.read_text(encoding="utf-8")
                data = yaml.safe_load(text)
                tpl = TemplateDefinition(**data)
                self._templates[tpl.name] = tpl

    def load_from_directory(self, path: Path) -> None:
        """Load template YAMLs from an arbitrary directory (for testing)."""
        for yaml_file in sorted(path.glob("*.yaml")):
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            tpl = TemplateDefinition(**data)
            self._templates[tpl.name] = tpl

    def get_template(self, name: str) -> TemplateDefinition | None:
        """Return the template for the given name, or None."""
        return self._templates.get(name)

    def list_all(self) -> list[TemplateDefinition]:
        """Return all loaded templates sorted by name."""
        return sorted(self._templates.values(), key=lambda t: t.name)

    @property
    def template_names(self) -> list[str]:
        """Return sorted list of loaded template names."""
        return sorted(self._templates.keys())
