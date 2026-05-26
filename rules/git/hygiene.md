# Git Hygiene

Use this rule for local git status logging, staging, commits, and rule cleanup that touches git-related commands or files.

## History Protection

- Do not delete git history, `.git/logs`, reflogs, local git stores, commit history, or git status-log artifacts unless the user explicitly asks for that deletion.
- Do not treat dated git-related files or logs as disposable just because they look old.
- If a cleanup target may remove local version-control history, stop and ask unless the user already gave a concrete deletion request.

## Staging Scope

- Do not force git usage when the user did not ask for it.
- Before staging or committing, inspect the staged scope.
- Omit high-storage, generated, private, or machine-specific sources unless the user explicitly wants them tracked.

Examples to avoid staging by default:

- raw PDFs
- videos
- datasets
- model weights
- checkpoints
- build products
- caches
- app bundles
- private secrets

## Broad Add Commands

Use broad staging commands carefully:

- `git add -A`
- `git add --all`
- `git add .`

When broad staging is necessary, check `git status --short` or equivalent first and exclude large/generated/private artifacts before committing.
