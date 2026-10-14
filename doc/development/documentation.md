# Build and publish documentation

The handbook uses Material for MkDocs. Documentation builds need Python only;
LLVM, MLIR and RISC-V hardware are not required.

```sh
python3 -m venv /tmp/weft-docs-env
. /tmp/weft-docs-env/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict --site-dir /tmp/weft-docs
python -m mkdocs serve
```

Open the local URL printed by the server. Build output belongs outside the
checkout; the default directory is `/tmp/weft-docs-site`, overridable through
`WEFT_DOCS_SITE_DIR` or `--site-dir`.

## Document structure

| Directory | Audience and responsibility |
|---|---|
| `getting-started/` | Installation and the first real compiled call |
| `programming-model/` | Author/compiler boundary and execution model |
| `dsl/` | Language constructs, storage, numerical semantics and canonical IR |
| `compiler/` | Physical machine, IR, passes, local operations and runtime contracts |
| `integrations/` | Practical application and framework entry points |
| `development/` | Hardware extension, numerical validation, measurement and documentation |

Use relative `.md` links within the handbook and GitHub links for source files.
When moving a page, update its incoming links and the navigation together. The
strict build checks page links and anchors. Keep runnable snippets aligned with
the public package and existing `examples/use.py` entry.

Stable design belongs in specifications; transient measurements, internal
instructions and historical work records are not part of the public handbook.

## Hosting

The Documentation workflow builds HTML for pull requests and main pushes. The
separate **Publish documentation** workflow deploys the current main branch to
GitHub Pages when explicitly dispatched. Configure the repository's Pages source
as **GitHub Actions**. Deployment uses the `github-pages` environment.

The configured public address is `https://tianchen-ai.github.io/Weft/`. A successful
local build alone does not establish that deployment completed; check the Pages
workflow and the resulting URL. Moving the repository requires updating
`site_url`, `repo_url`, package metadata and README links together.
