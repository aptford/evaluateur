# pr

## Overview

Create or update a GitHub Pull Request for the **current branch** using the GitHub CLI (`gh`).

This command should:

- Push the current branch to `origin` (set upstream if needed)
- Auto-detect the repo default base branch
- Create a PR if none exists for the branch, otherwise update the existing PR’s title/body
- Suggest (not auto-apply) labels/reviewers unless explicitly confirmed

## Steps

### 1) Preflight (must-run)

Run these commands and use their outputs to drive the rest of the workflow:

```bash
git status --porcelain
git branch --show-current
gh auth status
gh repo view --json defaultBranchRef --jq .defaultBranchRef.name
```

Rules:

- If `git status --porcelain` is non-empty, do not stage/commit anything automatically; proceed with PR creation/update based on the current working state.
- If `gh auth status` fails, stop and instruct the user to authenticate (`gh auth login`).

Set variables (conceptually):

- `BRANCH` = output of `git branch --show-current`
- `BASE` = output of `gh repo view ...defaultBranchRef.name`

### 2) Push current branch (must-run)

Ensure the branch exists on `origin` and has an upstream:

```bash
git push -u origin "$BRANCH"
```

If the upstream already exists, `git` will keep it; `-u` is safe.

### 3) Decide: update existing PR vs create a new PR

Check whether a PR already exists for the **current branch** (preferred), otherwise fall back to an explicit branch/head lookup.

Note: some `gh` versions do **not** support `gh pr view --head ...` (you’ll see `unknown flag: --head`). The patterns below are compatible:

```bash
# Preferred: current branch (works when you're on the branch already)
gh pr view --json number,url,title --jq '{number,url,title}' || \
# Fallback: pass branch name directly (supported by `gh pr view [<branch>]`)
gh pr view "$BRANCH" --json number,url,title --jq '{number,url,title}' || \
# Final fallback: list by head branch (useful if view-by-branch is unavailable)
gh pr list --head "$BRANCH" --json number,url,title --jq '.[0] // empty'
```

- If this succeeds, capture `PR_NUMBER` (or use the branch as a selector).
- If it fails (non-zero exit), treat it as “no PR exists yet.”

### 4) Generate PR title + body (agent output contract)

Produce EXACTLY the following output, in this order:

1. `TITLE:` a single-line PR title (≤ 72 chars preferred)
2. `BODY:` a markdown PR body using the template below
3. `SUGGESTED_LABELS:` comma-separated list (optional; can be empty)
4. `SUGGESTED_REVIEWERS:` comma-separated GitHub handles (optional; can be empty)
5. `APPLY_METADATA:` `yes` or `no` (must be `no` unless the user explicitly confirmed applying labels/reviewers)

Hard constraints (from GPT‑5.2 prompting guidance):

- Implement EXACTLY and ONLY what is asked. No extra work, no scope drift.
- Be concise: prefer bullets, avoid long narrative.
- Don’t fabricate links, issues, reviewers, or labels.

How to derive content:

- (Optional) Ensure base ref is up to date: `git fetch origin "$BASE" --prune`
- Use `git diff --stat origin/"$BASE"...HEAD` and `git log --oneline origin/"$BASE"..HEAD` (or equivalent) to summarize change scope.
- If UI changes, include “Screenshots” section but allow `N/A`.

How to suggest labels/reviewers (do NOT apply by default):

- Use `git diff --name-only origin/"$BASE"...HEAD` to see touched areas.
- Suggest labels that match repo conventions (only if you’re confident; otherwise leave empty).
- Suggest reviewers only if they are explicitly known from repo ownership conventions (CODEOWNERS) or prior context; otherwise leave empty.
- Set `APPLY_METADATA: no` unless the user explicitly confirms applying labels/reviewers.

### 5) Create or update the PR using `gh`

Write the body to a temp file (example uses heredoc; adjust as needed):

```bash
cat > /tmp/pr_body.md <<'EOF'
<paste BODY here>
EOF
```

Then:

- If PR exists (update):

```bash
gh pr edit "$BRANCH" --title "<paste TITLE>" --body-file /tmp/pr_body.md
```

- If PR does not exist (create):

```bash
gh pr create --base "$BASE" --head "$BRANCH" --title "<paste TITLE>" --body-file /tmp/pr_body.md
```

After creation, capture the PR number for optional metadata edits:

```bash
# Prefer current branch / branch selector (avoid `--head` for compatibility)
gh pr view "$BRANCH" --json number --jq .number
```

### 6) Optional: apply labels/reviewers (ONLY if explicitly confirmed)

Only run these if `APPLY_METADATA: yes`:

```bash
gh pr edit "$BRANCH" --add-label "<label1>" --add-label "<label2>"
gh pr edit "$BRANCH" --add-reviewer "<reviewer1>" --add-reviewer "<reviewer2>"
```

If not confirmed, keep them as suggestions in the PR body (or in the command output) but do not mutate PR metadata.

## PR Template

Use this template for the `BODY:` section:

```md
## Summary

- <1–3 bullets describing what changed>

## Why

- <1–2 bullets for motivation / context>

## What changed

- <bulleted list of key changes>

## Risk / rollout

- <1–2 bullets: risks, mitigations, rollout notes>
```
