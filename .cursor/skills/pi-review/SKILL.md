---
name: pi-review
description: Runs a read-only adversarial code review through `pi` (DeepSeek v4 flash) in a background terminal. Use when the user asks for a pi review, an adversarial review, or names this skill.
disable-model-invocation: true
---

# Pi review

Launch DeepSeek v4 flash via `pi` in a background terminal. Do not use Cursor subagents for this.

The reviewer has no shell. Dump the diffs and files yourself (`git show`, `git diff`) into a temp dir, write a short prompt that names those paths, and attach both with `@`.

```bash
pi -p --provider deepseek --model deepseek-v4-flash --thinking high \
  --tools read,grep,find,ls --no-session --no-extensions \
  --name adversarial-review \
  @<prompt> @<diff...>
```

Do not pass `bash`, `edit`, or `write`. If a review is already running, kill it before starting another.

Prompt the reviewer to be hostile: correctness, idiomaticness, simplicity. Findings first, each with file:line and why it matters, then a short judgment. No style nits unless they hide a bug. Do not frame the review as kill/keep.

When it finishes, read the output. Act on real defects; ignore noise.
