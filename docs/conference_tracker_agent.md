# Conference Tracker Agent

The script reads `Conferences`, searches the web for a target year, opens result
pages, verifies that the opened page itself mentions the target year and the
conference name, and appends only verified candidates to a separate review sheet.

It does not update the source sheet.

## Output Sheet

Default target sheet: `agent_candidates`.

The output header is:

1. all original columns from `Conferences`;
2. review fields:
   - `Human verified`
   - `agent_run_id`
   - `agent_checked_at`
   - `agent_source_row`
   - `agent_target_year`
   - `agent_confidence`
   - `agent_status`
   - `agent_evidence_url`
   - `agent_evidence_title`
   - `agent_evidence_snippet`
   - `agent_notes`

`Human verified` is written as a checkbox column.

The script refuses to write into a non-empty target sheet with a different
header. Use a clean sheet for a new schema instead of mixing old candidate rows.

## Search

The full run requires a Google SERP provider. The current default is Bright Data:

```bash
BRIGHTDATA_API_KEY=...
BRIGHTDATA_ZONE=serp_api
```

`SERPAPI_API_KEY` is also supported as a fallback provider. Without one of these
keys, the script stops before writing. `--allow-search-fallback`
exists only for debugging; do not use it for the cron run because HTML search is
slow, unstable, and not a reliable substitute for Google SERP results.

The script opens candidate result pages and rejects 404/missing-year pages.

## Dry Run

```bash
set -a; source .env; set +a

uv run python scripts/conference_tracker_agent.py \
  --spreadsheet-id 1GcA4jdTRlxM9THCLAam2Ix7g-ft__2R1VBrcsM0dRiU \
  --source-sheet Conferences \
  --target-sheet agent_candidates \
  --year 2027 \
  --limit 10 \
  --search-limit 5 \
  --dry-run
```

## Write Candidates

```bash
set -a; source .env; set +a

uv run python scripts/conference_tracker_agent.py \
  --spreadsheet-id 1GcA4jdTRlxM9THCLAam2Ix7g-ft__2R1VBrcsM0dRiU \
  --source-sheet Conferences \
  --target-sheet agent_candidates \
  --year 2027 \
  --limit 20 \
  --search-limit 5
```

After a write run, inspect `agent_candidates` directly. Do not treat the success
log as proof that the collected facts are analytically correct.

## Cron Example

```cron
0 9 * * MON cd /Users/gk/Code/super-duper-disser && set -a && . ./.env && set +a && uv run python scripts/conference_tracker_agent.py --spreadsheet-id 1GcA4jdTRlxM9THCLAam2Ix7g-ft__2R1VBrcsM0dRiU --source-sheet Conferences --target-sheet agent_candidates --year 2027 --limit 20 --search-limit 5 >> logs/conference_tracker_agent.log 2>&1
```

Create `logs/` before enabling cron.
