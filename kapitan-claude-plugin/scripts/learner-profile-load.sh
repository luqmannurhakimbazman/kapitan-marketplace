#!/bin/bash
# SessionStart hook: Load, validate, and sync-heal the learner profile.
# Stdout is injected into Claude's context at session start.

PROFILE="$HOME/.claude/leetcode-teacher-profile.md"
LEDGER="$HOME/.claude/leetcode-teacher-ledger.md"

# Extract session_id from hook input JSON (stdin)
INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
if [ -z "$SESSION_ID" ] || [ "$SESSION_ID" = "null" ]; then
  SESSION_ID="unknown"
fi

# --- First session: create both files with templates ---
if [ ! -f "$PROFILE" ]; then
  mkdir -p "$HOME/.claude"

  cat > "$PROFILE" << 'PROFILE_TEMPLATE'
## About Me

<!-- Write your goals, timeline, preferred language, and self-assessed level here. Claude reads this but won't overwrite without asking. -->

## Known Weaknesses

<!-- Active weaknesses tracked across sessions (max 10). -->

### Resolved

<!-- Weaknesses resolved long-term (max 10). -->

## Session History

<!-- Recent sessions, newest first (max 20). -->
PROFILE_TEMPLATE

  cat > "$LEDGER" << 'LEDGER_TEMPLATE'
# Session Ledger

| Timestamp | Session ID | Problem | Pattern | Mode | Verdict | Gaps | Review Due |
|-----------|------------|---------|---------|------|---------|------|------------|
LEDGER_TEMPLATE

  echo "=== SESSION METADATA ==="
  echo "Session ID: ${SESSION_ID}"
  echo "Session Timestamp: $(date +%Y-%m-%dT%H:%M)"
  echo "=== END SESSION METADATA ==="
  echo "=== LEARNER PROFILE (new — first session) ==="
  cat "$PROFILE"
  echo ""
  echo "=== END LEARNER PROFILE ==="
  echo "[FIRST SESSION] About Me is empty. Populate from observations during the session and confirm at end."
  exit 0
fi

# --- Profile exists: validate structure ---
REPAIRS=""

if ! grep -q "^## About Me" "$PROFILE" 2>/dev/null; then
  # Prepend About Me section
  TMPFILE=$(mktemp)
  echo '## About Me' > "$TMPFILE"
  echo '' >> "$TMPFILE"
  echo '<!-- Write your goals, timeline, preferred language, and self-assessed level here. -->' >> "$TMPFILE"
  echo '' >> "$TMPFILE"
  cat "$PROFILE" >> "$TMPFILE"
  mv "$TMPFILE" "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: About Me. Let the learner know if they didn't intend to remove it.\n"
fi

if ! grep -q "^## Known Weaknesses" "$PROFILE" 2>/dev/null; then
  echo '' >> "$PROFILE"
  echo '## Known Weaknesses' >> "$PROFILE"
  echo '' >> "$PROFILE"
  echo '### Resolved' >> "$PROFILE"
  echo '' >> "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: Known Weaknesses. Let the learner know if they didn't intend to remove it.\n"
fi

if ! grep -q "^## Session History" "$PROFILE" 2>/dev/null; then
  echo '' >> "$PROFILE"
  echo '## Session History' >> "$PROFILE"
  echo '' >> "$PROFILE"
  REPAIRS="${REPAIRS}[REPAIRED] Restored missing section: Session History. Let the learner know if they didn't intend to remove it.\n"
fi

# --- Self-heal sync: check profile vs ledger consistency ---
# Get last session history entry timestamp from profile
LAST_PROFILE_TS=$(grep -E '^### [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2} \|' "$PROFILE" 2>/dev/null | head -1 | sed 's/^### //' | cut -d'|' -f1 | xargs)

# Get last ledger row timestamp
LAST_LEDGER_TS=$(tail -1 "$LEDGER" 2>/dev/null | grep -E '^\|' | sed 's/^| *//' | cut -d'|' -f1 | xargs)

if [ -n "$LAST_PROFILE_TS" ] && [ -n "$LAST_LEDGER_TS" ]; then
  # Check if profile has a newer entry than ledger
  if [[ "$LAST_PROFILE_TS" > "$LAST_LEDGER_TS" ]]; then
    # Profile has entry not in ledger — extract fields and append
    ENTRY_LINE=$(grep -E "^### ${LAST_PROFILE_TS} \|" "$PROFILE" 2>/dev/null | head -1)
    P_PROBLEM=$(echo "$ENTRY_LINE" | cut -d'|' -f2 | xargs)
    P_MODE=$(echo "$ENTRY_LINE" | cut -d'|' -f3 | xargs)
    P_VERDICT=$(echo "$ENTRY_LINE" | cut -d'|' -f4 | xargs)

    # Get gaps and review from lines following the entry
    ENTRY_LINE_NUM=$(grep -n "^### ${LAST_PROFILE_TS} |" "$PROFILE" | head -1 | cut -d: -f1)
    if [ -n "$ENTRY_LINE_NUM" ]; then
      GAPS_LINE_NUM=$((ENTRY_LINE_NUM + 1))
      REVIEW_LINE_NUM=$((ENTRY_LINE_NUM + 2))
      P_GAPS=$(sed -n "${GAPS_LINE_NUM}p" "$PROFILE" | sed 's/^Gaps: //')
      P_REVIEW=$(sed -n "${REVIEW_LINE_NUM}p" "$PROFILE" | sed 's/^Review: //')
    fi

    [ -z "$P_GAPS" ] && P_GAPS="none"
    [ -z "$P_REVIEW" ] && P_REVIEW="—"

    echo "| ${LAST_PROFILE_TS} | sync-heal | ${P_PROBLEM} | unknown | ${P_MODE} | ${P_VERDICT} | ${P_GAPS} | ${P_REVIEW} |" >> "$LEDGER"
    REPAIRS="${REPAIRS}[SYNC] Appended missing ledger row for ${LAST_PROFILE_TS} session (pattern=unknown).\n"
  fi
elif [ -n "$LAST_PROFILE_TS" ] && [ -z "$LAST_LEDGER_TS" ]; then
  # Profile has entries but ledger is empty (header only)
  ENTRY_LINE=$(grep -E "^### ${LAST_PROFILE_TS} \|" "$PROFILE" 2>/dev/null | head -1)
  P_PROBLEM=$(echo "$ENTRY_LINE" | cut -d'|' -f2 | xargs)
  P_MODE=$(echo "$ENTRY_LINE" | cut -d'|' -f3 | xargs)
  P_VERDICT=$(echo "$ENTRY_LINE" | cut -d'|' -f4 | xargs)

  ENTRY_LINE_NUM=$(grep -n "^### ${LAST_PROFILE_TS} |" "$PROFILE" | head -1 | cut -d: -f1)
  if [ -n "$ENTRY_LINE_NUM" ]; then
    GAPS_LINE_NUM=$((ENTRY_LINE_NUM + 1))
    REVIEW_LINE_NUM=$((ENTRY_LINE_NUM + 2))
    P_GAPS=$(sed -n "${GAPS_LINE_NUM}p" "$PROFILE" | sed 's/^Gaps: //')
    P_REVIEW=$(sed -n "${REVIEW_LINE_NUM}p" "$PROFILE" | sed 's/^Review: //')
  fi

  [ -z "$P_GAPS" ] && P_GAPS="none"
  [ -z "$P_REVIEW" ] && P_REVIEW="—"

  echo "| ${LAST_PROFILE_TS} | sync-heal | ${P_PROBLEM} | unknown | ${P_MODE} | ${P_VERDICT} | ${P_GAPS} | ${P_REVIEW} |" >> "$LEDGER"
  REPAIRS="${REPAIRS}[SYNC] Appended missing ledger row for ${LAST_PROFILE_TS} session (pattern=unknown).\n"
fi

# --- Output session metadata ---
echo "=== SESSION METADATA ==="
echo "Session ID: ${SESSION_ID}"
echo "Session Timestamp: $(date +%Y-%m-%dT%H:%M)"
echo "=== END SESSION METADATA ==="

# --- Output full profile ---
echo "=== LEARNER PROFILE (loaded automatically at session start) ==="
cat "$PROFILE"
echo ""
echo "=== END LEARNER PROFILE ==="

# --- Output repairs if any ---
if [ -n "$REPAIRS" ]; then
  echo ""
  echo -e "$REPAIRS"
fi

# --- Retest suggestions for resolved (short-term) weaknesses ---
RETEST_OUTPUT=""
while IFS= read -r line; do
  # Extract the label and find the Last tested timestamp
  LABEL=$(echo "$line" | sed 's/^- \*\*\(.*\)\*\*.*/\1/')

  # Look for Last tested in the next few lines after this entry in the profile
  LINE_NUM=$(grep -n "^- \*\*${LABEL}\*\*" "$PROFILE" | head -1 | cut -d: -f1)
  if [ -n "$LINE_NUM" ]; then
    # Search the next 3 lines for Last tested
    LAST_TESTED=$(sed -n "$((LINE_NUM+1)),$((LINE_NUM+3))p" "$PROFILE" | grep -o 'Last tested: [0-9T:-]*' | sed 's/Last tested: //')

    if [ -n "$LAST_TESTED" ]; then
      # Compare dates (strip time for date arithmetic)
      TESTED_DATE=$(echo "$LAST_TESTED" | cut -dT -f1)
      TODAY=$(date +%Y-%m-%d)

      # Calculate weeks difference using portable date arithmetic
      if command -v python3 &>/dev/null; then
        WEEKS_AGO=$(python3 -c "
from datetime import datetime
d1 = datetime.strptime('${TESTED_DATE}', '%Y-%m-%d')
d2 = datetime.strptime('${TODAY}', '%Y-%m-%d')
print((d2-d1).days // 7)
" 2>/dev/null)
      else
        WEEKS_AGO=0
      fi

      if [ -n "$WEEKS_AGO" ] && [ "$WEEKS_AGO" -ge 2 ] 2>/dev/null; then
        RETEST_OUTPUT="${RETEST_OUTPUT}- ${LABEL} (last tested ${LAST_TESTED}, ${WEEKS_AGO} weeks ago)\n"
      fi
    fi
  fi
done < <(grep "resolved (short-term)" "$PROFILE" | grep "^- \*\*")

if [ -n "$RETEST_OUTPUT" ]; then
  echo ""
  echo "=== RETEST SUGGESTIONS ==="
  echo -e "$RETEST_OUTPUT"
  echo "=== END RETEST SUGGESTIONS ==="
fi

exit 0
