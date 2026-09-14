# Working with ReprForge

- Start with README.md and docs/architecture.md.
- Organize implementation by component and experiments by evaluation purpose.
- Keep research diaries, historical outputs and paper figure sources outside Git.
- Keep datasets, weights, retained states and generated results outside Git.
- Reproduction code must run from public inputs and documented configurations.
- Use the A100 protocol for paper comparisons; preserve reference and timing scope.
- Validate with Ruff, pytest, CPU examples and a wheel build. Smoke-check CLI help.
- Do not launch GPU jobs or publish changes without user authorization.
