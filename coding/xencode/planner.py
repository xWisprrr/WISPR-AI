"""Generates a project file plan from an XencodeSpec."""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from llm.router import LLMRouter, TaskType
from coding.xencode.schemas import PlannedFile, XencodePlan, XencodeSpec

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """\
You are an expert software architect. Generate a complete, working file plan for the project.

Return ONLY valid JSON with this exact structure:
{
  "files": [
    {"path": "relative/path/file.ext", "content": "...full file content...", "description": "brief description"}
  ]
}

Rules:
- Always include README.md with full documentation
- Always include the entry point file with real, working code
- Generate complete code — no placeholders, no "TODO", no "..."
- For Python: include requirements.txt if there are dependencies
- For JavaScript/TypeScript: include package.json
- For Go: include go.mod
- For Rust: include Cargo.toml
- Keep all code production-quality and working
"""


class Planner:
    """Generates a file plan from an XencodeSpec."""

    def __init__(self, llm: Optional[LLMRouter] = None) -> None:
        self.llm = llm or LLMRouter()

    async def plan(self, spec: XencodeSpec, workspace: str) -> XencodePlan:
        """Generate a file plan for *spec*. Falls back to minimal plan on LLM failure."""
        try:
            return await self._plan_with_llm(spec, workspace)
        except Exception as exc:
            logger.warning("LLM plan generation failed (%s), using fallback.", exc)
            return self._fallback_plan(spec, workspace)

    async def _plan_with_llm(self, spec: XencodeSpec, workspace: str) -> XencodePlan:
        spec_summary = (
            f"Project: {spec.project_name}\n"
            f"Language: {spec.language}\n"
            f"Description: {spec.description}\n"
            f"Requirements: {', '.join(spec.requirements) or 'none'}\n"
            f"Entry point: {spec.entry_point}\n"
            f"Build command: {spec.build_command or 'none'}\n"
        )
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": f"Generate a complete project for:\n\n{spec_summary}"},
        ]
        raw = await self.llm.complete(messages, task_type=TaskType.CODING, temperature=0.2)
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        files = [
            PlannedFile(
                path=f["path"],
                content=f["content"],
                description=f.get("description", ""),
            )
            for f in data.get("files", [])
        ]
        readme_included = any(f.path.lower() in ("readme.md", "./readme.md") for f in files)
        return XencodePlan(
            spec=spec, files=files, workspace=workspace, readme_included=readme_included
        )

    def _fallback_plan(self, spec: XencodeSpec, workspace: str) -> XencodePlan:
        """Minimal working project without LLM."""
        files = _generate_fallback_files(spec)
        readme_included = any(f.path.lower() == "readme.md" for f in files)
        return XencodePlan(
            spec=spec, files=files, workspace=workspace, readme_included=readme_included
        )


# ── fallback file generators ──────────────────────────────────────────────────

def _generate_fallback_files(spec: XencodeSpec) -> List[PlannedFile]:
    """Generate a minimal set of project files without LLM assistance."""
    files: List[PlannedFile] = []
    lang = spec.language

    files.append(PlannedFile(path="README.md", content=_readme(spec), description="Project README"))

    if lang == "python":
        files.extend(_python_files(spec))
    elif lang == "javascript":
        files.extend(_js_files(spec, ext="js"))
    elif lang == "typescript":
        files.extend(_js_files(spec, ext="ts"))
    elif lang == "go":
        files.extend(_go_files(spec))
    elif lang == "rust":
        files.extend(_rust_files(spec))
    elif lang == "java":
        files.extend(_java_files(spec))
    elif lang == "cpp":
        files.extend(_cpp_files(spec))
    else:
        files.append(
            PlannedFile(
                path=spec.entry_point,
                content=f"# {spec.project_name}\n# {spec.description}\n",
                description="Entry point",
            )
        )
    return files


def _readme(spec: XencodeSpec) -> str:
    run_cmd = {
        "python": f"python {spec.entry_point}",
        "javascript": f"node {spec.entry_point}",
        "typescript": f"npx ts-node {spec.entry_point}",
        "go": "go run main.go",
        "rust": "cargo run",
        "java": "java Main",
        "cpp": "./app",
        "bash": f"bash {spec.entry_point}",
    }.get(spec.language, f"./{spec.entry_point}")

    setup = f"```\n{spec.build_command}\n```" if spec.build_command else "_No build step required._"
    return (
        f"# {spec.project_name}\n\n"
        f"{spec.description}\n\n"
        f"## Language\n\n{spec.language}\n\n"
        f"## Setup\n\n{setup}\n\n"
        f"## Usage\n\n```\n{run_cmd}\n```\n"
    )


def _python_files(spec: XencodeSpec) -> List[PlannedFile]:
    main = (
        f'"""{spec.project_name} — {spec.description}"""\n\n\n'
        f"def main() -> None:\n"
        f'    print("Hello from {spec.project_name}")\n\n\n'
        f'if __name__ == "__main__":\n'
        f"    main()\n"
    )
    files = [PlannedFile(path="main.py", content=main, description="Entry point")]
    if spec.requirements:
        files.append(
            PlannedFile(
                path="requirements.txt",
                content="\n".join(spec.requirements) + "\n",
                description="Python dependencies",
            )
        )
    return files


def _js_files(spec: XencodeSpec, ext: str) -> List[PlannedFile]:
    main = f"// {spec.project_name} — {spec.description}\nconsole.log('Hello from {spec.project_name}');\n"
    pkg = json.dumps(
        {
            "name": spec.project_name,
            "version": "1.0.0",
            "description": spec.description,
            "main": f"index.{ext}",
            "scripts": {"start": f"node index.{ext}", "test": "echo \"no tests\""},
            "dependencies": {r: "latest" for r in spec.requirements},
        },
        indent=2,
    )
    return [
        PlannedFile(path=f"index.{ext}", content=main, description="Entry point"),
        PlannedFile(path="package.json", content=pkg, description="Package manifest"),
    ]


def _go_files(spec: XencodeSpec) -> List[PlannedFile]:
    main = (
        "package main\n\nimport \"fmt\"\n\n"
        f'func main() {{\n\tfmt.Println("Hello from {spec.project_name}")\n}}\n'
    )
    mod = f"module {spec.project_name}\n\ngo 1.21\n"
    return [
        PlannedFile(path="main.go", content=main, description="Entry point"),
        PlannedFile(path="go.mod", content=mod, description="Go module"),
    ]


def _rust_files(spec: XencodeSpec) -> List[PlannedFile]:
    main = f'fn main() {{\n    println!("Hello from {spec.project_name}");\n}}\n'
    cargo = (
        f'[package]\nname = "{spec.project_name}"\n'
        f'version = "0.1.0"\nedition = "2021"\n'
    )
    return [
        PlannedFile(path="src/main.rs", content=main, description="Entry point"),
        PlannedFile(path="Cargo.toml", content=cargo, description="Cargo manifest"),
    ]


def _java_files(spec: XencodeSpec) -> List[PlannedFile]:
    main = (
        f"public class Main {{\n"
        f'    public static void main(String[] args) {{\n'
        f'        System.out.println("Hello from {spec.project_name}");\n'
        f"    }}\n}}\n"
    )
    return [PlannedFile(path="Main.java", content=main, description="Entry point")]


def _cpp_files(spec: XencodeSpec) -> List[PlannedFile]:
    main = (
        "#include <iostream>\n\nint main() {\n"
        f'    std::cout << "Hello from {spec.project_name}" << std::endl;\n'
        "    return 0;\n}\n"
    )
    return [PlannedFile(path="main.cpp", content=main, description="Entry point")]
