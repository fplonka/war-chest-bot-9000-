---
name: pi-review
description: Adversarial review via `pi` (DeepSeek v4 flash). Use when the user asks for a pi review, an adversarial review, or after landing a commit or starting a training run.
---

Launch DeepSeek v4 flash with `pi`. Do not use Cursor subagents. The reviewer has no shell: dump the diffs yourself and attach them with `@`. Do not pass `bash`, `edit`, or `write`. Kill a review that is already running first.

```bash
pi -p --provider deepseek --model deepseek-v4-flash --thinking high \
  --tools read,grep,find,ls --no-session --no-extensions \
  --name adversarial-review \
  @PROMPT @DIFF...
```

PROMPT asks for a hostile review (correctness, idiomaticness, simplicity), findings first with file:line, then a heading exactly `## Judgment`. No style nits unless they hide a bug.

Wake on finish: `block_until_ms: 0` and `notify_on_output` with `pattern: ## Judgment`, `reason: pi review finished`. Same for `tools/box.sh go` / `follow` with `pattern: ^JOB_DONE`, `reason: box job finished`. Ending the turn is what delivers the wake.
