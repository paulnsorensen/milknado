from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_FILE = ROOT / "integrations" / "spec" / "commands.yaml"
TEMPLATE_DIR = ROOT / "integrations" / "spec" / "templates"
CURSOR_COMMANDS_DIR = ROOT / "integrations" / "cursor-milknado" / "commands"
GEMINI_COMMANDS_DIR = ROOT / "integrations" / "gemini-milknado" / "commands"


def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fill_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _body_block(value: str) -> str:
    if not value.strip():
        return ""
    return f"\n{value}\n"


def _note_block(value: str) -> str:
    if not value.strip():
        return ""
    return f"\n{value}"


def _render_cursor(command: dict[str, str], template: str) -> str:
    values = {
        "id": command["id"],
        "title": command["title"],
        "cursor_description": command["cursor_description"],
        "cursor_command": command["cursor_command"],
        "cursor_body_block": _body_block(command.get("cursor_body_intro", "")),
        "cursor_note_block": _note_block(command.get("cursor_note", "")),
    }
    rendered = _fill_template(template, values)
    return rendered.rstrip() + "\n"


def _render_gemini(command: dict[str, str], template: str) -> str:
    values = {
        "gemini_description": command["gemini_description"].replace('"', '\\"'),
        "gemini_heading": command["gemini_heading"],
        "gemini_prompt_body": command["gemini_prompt_body"].rstrip(),
    }
    rendered = _fill_template(template, values)
    return rendered.rstrip() + "\n"


def main() -> None:
    spec = yaml.safe_load(SPEC_FILE.read_text(encoding="utf-8"))
    commands: list[dict[str, str]] = spec["commands"]

    cursor_template = _load_template(TEMPLATE_DIR / "cursor-command.md.tmpl")
    gemini_template = _load_template(TEMPLATE_DIR / "gemini-command.toml.tmpl")

    CURSOR_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    GEMINI_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)

    for command in commands:
        cursor_path = CURSOR_COMMANDS_DIR / f"{command['id']}.md"
        cursor_path.write_text(_render_cursor(command, cursor_template), encoding="utf-8")

        gemini_path = GEMINI_COMMANDS_DIR / command["gemini_file"]
        gemini_path.write_text(_render_gemini(command, gemini_template), encoding="utf-8")

    print(f"Generated {len(commands)} cursor and gemini command files.")


if __name__ == "__main__":
    main()
