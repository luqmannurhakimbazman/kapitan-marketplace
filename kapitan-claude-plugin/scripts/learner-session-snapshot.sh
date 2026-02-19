#!/bin/bash
# PreCompact hook: Write a minimal procedural reminder before context compaction.
# No transcript parsing, no semantic extraction — just a procedural nudge.

SCRATCHPAD="$HOME/.claude/leetcode-session-state.md"
INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)

# Only write if this looks like a leetcode session
if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ] && grep -q "leetcode-teacher" "$TRANSCRIPT" 2>/dev/null; then
  SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
  if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
    SESSION_ID="unknown"
  fi

  EXISTING_TS=""
  if [ -f "$SCRATCHPAD" ]; then
    EXISTING_TS=$(grep -o 'Session Timestamp: [0-9T:-]*' "$SCRATCHPAD" | sed 's/Session Timestamp: //')
  fi
  SESSION_TS="${EXISTING_TS:-$(date +%Y-%m-%dT%H:%M)}"

  cat > "$SCRATCHPAD" << EOF
# LeetCode Session In Progress (saved before compaction)
- You are in a leetcode-teacher session. Read ~/.claude/leetcode-teacher-profile.md for context.
- Session ID: ${SESSION_ID}
- Session Timestamp: ${SESSION_TS}
- Write-back required at session end: Step 8B (learning) or R7B (recall).
- Write ledger row first, then profile entry.
EOF
fi

exit 0
