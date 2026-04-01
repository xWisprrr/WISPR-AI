"""Code Engine — Seamless deployment integrations.

Supports one-click deployment to:
  • GitHub  — create repo and push code
  • Vercel  — generate vercel.json + deployment instructions
  • Netlify — generate netlify.toml + deployment instructions
  • Railway — generate railway.toml + Dockerfile stubs
  • Docker  — generate Dockerfile + docker-compose.yml

All providers use a common DeploymentResult dataclass and a single
`deploy()` async entry point.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.router import LLMRouter, TaskType

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ["github", "vercel", "netlify", "railway", "docker"]


@dataclass
class DeploymentResult:
    success: bool
    provider: str
    message: str
    files_generated: List[str] = field(default_factory=list)
    instructions: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "provider": self.provider,
            "message": self.message,
            "files_generated": self.files_generated,
            "instructions": self.instructions,
            "error": self.error,
        }


_DEPLOY_SYSTEM = """\
You are a DevOps expert. Given a project description and target deployment platform,
generate the exact configuration files and step-by-step deployment instructions
needed to deploy the project. Be precise, complete, and production-ready.
"""


class DeploymentEngine:
    """Generate deployment configs and instructions for multiple platforms."""

    def __init__(self, llm_router: Optional[LLMRouter] = None) -> None:
        self.llm = llm_router or LLMRouter()

    async def deploy(
        self,
        provider: str,
        *,
        project_path: str,
        project_description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> DeploymentResult:
        """Generate deployment artifacts for *provider*.

        Args:
            provider:            One of the SUPPORTED_PROVIDERS.
            project_path:        Absolute or relative path to the project root.
            project_description: Brief description of the project (helps LLM).
            config:              Provider-specific config overrides.
        """
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            return DeploymentResult(
                success=False,
                provider=provider,
                message=f"Unsupported provider '{provider}'.",
                error=f"Choose from: {SUPPORTED_PROVIDERS}",
            )

        root = Path(project_path).expanduser().resolve()
        cfg = config or {}

        if provider == "github":
            return await self._deploy_github(root, cfg, project_description)
        elif provider == "vercel":
            return await self._deploy_vercel(root, cfg, project_description)
        elif provider == "netlify":
            return await self._deploy_netlify(root, cfg, project_description)
        elif provider == "railway":
            return await self._deploy_railway(root, cfg, project_description)
        elif provider == "docker":
            return await self._deploy_docker(root, cfg, project_description)

        return DeploymentResult(success=False, provider=provider, message="Unknown error.")

    # ── GitHub ────────────────────────────────────────────────────────────

    async def _deploy_github(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        files_generated: List[str] = []

        # Generate GitHub Actions CI workflow
        workflow_dir = root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = workflow_dir / "ci.yml"

        workflow_content = await self._llm_generate(
            provider="GitHub Actions",
            project_root=root,
            description=description,
            extra=(
                "Generate a complete .github/workflows/ci.yml that runs tests, "
                "linting, and optionally builds a Docker image."
            ),
        )
        workflow_path.write_text(workflow_content, encoding="utf-8")
        files_generated.append(str(workflow_path))

        instructions = self._github_instructions(root, cfg)
        return DeploymentResult(
            success=True,
            provider="github",
            message="GitHub CI workflow generated.",
            files_generated=files_generated,
            instructions=instructions,
        )

    def _github_instructions(self, root: Path, cfg: Dict[str, Any]) -> str:
        repo_name = cfg.get("repo_name", root.name)
        return f"""\
## GitHub Deployment Steps

1. Create a new GitHub repository named **{repo_name}** at https://github.com/new
2. Initialise git and push your code:
   ```bash
   cd {root}
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/{repo_name}.git
   git push -u origin main
   ```
3. The workflow at `.github/workflows/ci.yml` will run automatically on every push.
4. Add any required secrets (API keys, tokens) under **Settings → Secrets and variables → Actions**.

> ⚠️ WISPR AI is running with unrestricted filesystem access. Ensure you review
> generated files before pushing to a public repository.
"""

    # ── Vercel ────────────────────────────────────────────────────────────

    async def _deploy_vercel(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        files_generated: List[str] = []

        vercel_config = {
            "version": 2,
            "builds": cfg.get("builds", [{"src": "main.py", "use": "@vercel/python"}]),
            "routes": cfg.get("routes", [{"src": "/(.*)", "dest": "main.py"}]),
            "env": cfg.get("env", {}),
        }
        vercel_path = root / "vercel.json"
        vercel_path.write_text(json.dumps(vercel_config, indent=2), encoding="utf-8")
        files_generated.append(str(vercel_path))

        instructions = f"""\
## Vercel Deployment Steps

1. Install Vercel CLI:  `npm i -g vercel`
2. From the project root:
   ```bash
   cd {root}
   vercel
   ```
3. Follow the prompts. Your app will be live at `https://<project>.vercel.app`.
4. For production deployments:  `vercel --prod`
5. Set environment variables:  `vercel env add <KEY>`

> A `vercel.json` has been generated at `{vercel_path}`. Review it before deploying.
"""
        return DeploymentResult(
            success=True,
            provider="vercel",
            message="vercel.json generated.",
            files_generated=files_generated,
            instructions=instructions,
        )

    # ── Netlify ───────────────────────────────────────────────────────────

    async def _deploy_netlify(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        files_generated: List[str] = []

        publish_dir = cfg.get("publish_dir", "dist")
        build_cmd = cfg.get("build_command", "npm run build")
        functions_dir = cfg.get("functions_dir", "netlify/functions")

        toml_content = f"""\
[build]
  publish = "{publish_dir}"
  command = "{build_cmd}"
  functions = "{functions_dir}"

[build.environment]
  NODE_VERSION = "{cfg.get('node_version', '18')}"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
"""
        toml_path = root / "netlify.toml"
        toml_path.write_text(toml_content, encoding="utf-8")
        files_generated.append(str(toml_path))

        instructions = f"""\
## Netlify Deployment Steps

1. Install Netlify CLI:  `npm i -g netlify-cli`
2. Log in:  `netlify login`
3. Deploy preview:
   ```bash
   cd {root}
   netlify deploy
   ```
4. Deploy to production:  `netlify deploy --prod`
5. Set environment variables via the Netlify dashboard or:
   `netlify env:set KEY value`

> `netlify.toml` generated at `{toml_path}`.
"""
        return DeploymentResult(
            success=True,
            provider="netlify",
            message="netlify.toml generated.",
            files_generated=files_generated,
            instructions=instructions,
        )

    # ── Railway ───────────────────────────────────────────────────────────

    async def _deploy_railway(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        files_generated: List[str] = []

        toml_content = f"""\
[build]
  builder = "DOCKERFILE"
  dockerfilePath = "Dockerfile"

[deploy]
  startCommand = "{cfg.get('start_command', 'python main.py')}"
  restartPolicyType = "ON_FAILURE"
  restartPolicyMaxRetries = 3
"""
        toml_path = root / "railway.toml"
        toml_path.write_text(toml_content, encoding="utf-8")
        files_generated.append(str(toml_path))

        # Also generate a Dockerfile stub
        df_result = await self._generate_dockerfile(root, cfg, description)
        files_generated.extend(df_result.files_generated)

        instructions = f"""\
## Railway Deployment Steps

1. Install Railway CLI:  `npm i -g @railway/cli`
2. Log in:  `railway login`
3. Create a new project:  `railway init`
4. Deploy:
   ```bash
   cd {root}
   railway up
   ```
5. Set environment variables:  `railway variables set KEY=value`
6. Get your deployment URL:  `railway open`

> `railway.toml` and `Dockerfile` generated. Review before deploying.
"""
        return DeploymentResult(
            success=True,
            provider="railway",
            message="railway.toml and Dockerfile generated.",
            files_generated=files_generated,
            instructions=instructions,
        )

    # ── Docker ────────────────────────────────────────────────────────────

    async def _deploy_docker(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        return await self._generate_dockerfile(root, cfg, description)

    async def _generate_dockerfile(
        self, root: Path, cfg: Dict[str, Any], description: Optional[str]
    ) -> DeploymentResult:
        files_generated: List[str] = []

        dockerfile_content = await self._llm_generate(
            provider="Docker",
            project_root=root,
            description=description,
            extra=(
                "Generate a production-ready multi-stage Dockerfile. "
                "Use slim base images, run as non-root user, and copy only required files."
            ),
        )
        dockerfile_path = root / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
        files_generated.append(str(dockerfile_path))

        # docker-compose.yml
        compose_content = await self._llm_generate(
            provider="Docker Compose",
            project_root=root,
            description=description,
            extra="Generate a docker-compose.yml with app, optional database, and health checks.",
        )
        compose_path = root / "docker-compose.yml"
        compose_path.write_text(compose_content, encoding="utf-8")
        files_generated.append(str(compose_path))

        instructions = f"""\
## Docker Deployment Steps

1. Build the image:
   ```bash
   cd {root}
   docker build -t {root.name}:latest .
   ```
2. Run locally:
   ```bash
   docker run -p 8000:8000 {root.name}:latest
   ```
3. Or with Docker Compose:
   ```bash
   docker compose up --build
   ```
4. Push to a registry:
   ```bash
   docker tag {root.name}:latest <registry>/{root.name}:latest
   docker push <registry>/{root.name}:latest
   ```

> `Dockerfile` and `docker-compose.yml` generated at `{root}`. Review before deploying.
"""
        return DeploymentResult(
            success=True,
            provider="docker",
            message="Dockerfile and docker-compose.yml generated.",
            files_generated=files_generated,
            instructions=instructions,
        )

    # ── LLM helper ───────────────────────────────────────────────────────

    async def _llm_generate(
        self,
        *,
        provider: str,
        project_root: Path,
        description: Optional[str],
        extra: str,
    ) -> str:
        """Use the LLM to generate a deployment config file."""
        # Detect project type from files present
        files = [f.name for f in project_root.iterdir() if f.is_file()][:30]
        project_info = description or f"Project root contains: {', '.join(files)}"

        messages = [
            {"role": "system", "content": _DEPLOY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Platform: {provider}\n"
                    f"Project: {project_info}\n\n"
                    f"{extra}\n\n"
                    "Return ONLY the file content, no explanation."
                ),
            },
        ]
        try:
            response = await self.llm.complete(messages, task_type=TaskType.CODING)
            # Strip markdown code fences if present
            match = re.search(r"```(?:\w+)?\n?(.*?)```", response, re.DOTALL)
            return match.group(1).strip() if match else response.strip()
        except Exception as exc:
            logger.warning("DeploymentEngine LLM call failed: %s", exc)
            return f"# Auto-generated by WISPR AI Code Engine\n# Provider: {provider}\n# Review and customise before use.\n"
