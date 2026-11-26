#!/usr/bin/env python3
"""Generate .opencode/agents/*.md from runtime-agnostic roles/*.yaml."""

import json
from pathlib import Path
from typing import Any, cast

import jsonschema
import yaml

ROOT = Path(__file__).parent.parent
ROLES_DIR = ROOT / "roles"
AGENTS_DIR = ROOT / ".opencode" / "agents"
SCHEMA_PATH = ROLES_DIR / "role-schema.json"


def load_schema() -> dict[str, Any]:
    """Load the JSON Schema used to validate role definitions."""
    return cast(dict[str, Any], json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def validate_role(data: dict[str, Any], source: str, schema: dict[str, Any]) -> None:
    """Validate a role dict against the JSON Schema."""
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        path = " -> ".join(str(part) for part in exc.path) or "<root>"
        raise ValueError(
            f"Validation error in {source} at {path}: {exc.message}"
        ) from exc


def load_role(path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    """Load and validate a single role YAML file."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Role file {path} must contain a mapping")
    data: dict[str, Any] = raw
    validate_role(data, source=path.name, schema=schema)
    return data


def build_bash_permission(role: dict[str, Any]) -> dict[str, str]:
    """Build OpenCode permission.bash from shell pattern lists."""
    bash: dict[str, str] = {}
    bash["*"] = role["default_shell_action"]
    for pattern in role.get("ask_shell_patterns", []):
        bash[pattern] = "ask"
    for pattern in role["allowed_shell_patterns"]:
        bash[pattern] = "allow"
    for pattern in role.get("denied_shell_patterns", []):
        bash[pattern] = "deny"
    return bash


def build_frontmatter(role: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenCode frontmatter for a role."""
    return {
        "description": f"{role['name']}. {role['description']}",
        "mode": role["mode"],
        "permission": {
            "edit": role["permissions"]["edit"],
            "bash": build_bash_permission(role),
        },
    }


def render_agent(role: dict[str, Any]) -> str:
    """Render the complete agent Markdown file for a role."""
    frontmatter = build_frontmatter(role)
    frontmatter_yaml = yaml.dump(
        frontmatter,
        allow_unicode=True,
        sort_keys=False,
        width=float("inf"),
        default_flow_style=False,
    )
    instructions: str = role["instructions"]
    return f"---\n{frontmatter_yaml}---\n\n{instructions}\n"


def generate_all() -> None:
    """Regenerate all .opencode/agents/*.md files from roles/*.yaml."""
    schema = load_schema()
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(ROLES_DIR.glob("*.yaml")):
        role = load_role(path, schema)
        agent_path = AGENTS_DIR / f"{role['id']}.md"
        agent_path.write_text(render_agent(role), encoding="utf-8")
        print(f"Generated {agent_path.relative_to(ROOT)}")


def main() -> None:
    """CLI entry point."""
    generate_all()


if __name__ == "__main__":
    main()
