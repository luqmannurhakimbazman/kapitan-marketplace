---
description: Diagnose and fix bugs in Claude Skills by tracing execution paths and identifying root causes
argument-hint: <skill-path> <bug description>
---

# /debug

Diagnose a bug in a Claude Skill. Do not guess. Trace the code.

## Usage

```
/debug <skill-path> <bug description>
```

If no arguments are provided, ask for the skill path and a description of the bug.

## Input Parsing

Parse `$ARGUMENTS` as follows:
- First argument: path to the skill directory (e.g., `./skills/leetcode-teacher` or `~/.claude/skills/my-skill`)
- Remaining arguments: natural language bug description

If the path is ambiguous, search `.claude/skills/`, `~/.claude/skills/`, and the current directory for a matching skill name.

## Phase 0: Load Context

1. Read `SKILL.md` first.
2. For skills with many reference files, identify references relevant to the bug from SKILL.md, then read only those. For small skills (fewer than 5 files total), read everything.
3. Build a mental map of the skill's intended workflow before looking for bugs.

## Phase 1: Reproduce & Locate

Identify the **expected behavior** and **actual behavior** from the bug description.

Then find both code paths:

**Working path** (the behavior that IS correct):
- File, line numbers, trigger conditions
- Why this works

**Broken path** (the behavior that is NOT correct):
- File, line numbers, trigger conditions
- If the path doesn't exist, that IS the bug — document what's missing and where it should be

If the bug description only describes broken behavior without a working counterpart, map the broken path and check it against the skill's stated workflow.

## Phase 2: Diff the Paths

If both paths exist, answer:

1. Are both triggered by the same event, or do they have different triggers?
2. Is the broken path conditional on something the working path is not?
3. Is there an ordering dependency? Could a failure in sequencing cause it to be skipped?
4. Is there a naming inconsistency? (e.g., "ledger" in one place, "log" in another)
5. Is the broken path's instruction ambiguous enough to be interpreted as optional?
6. Is the file path or reference target correct?

## Phase 3: Check Instruction Failure Patterns

Check every pattern below against the broken behavior. Flag all matches.

### 1. Buried Instruction

The instruction exists but is approximately past line 300 in SKILL.md. As Claude's context fills with user conversation, late instructions get deprioritized.

**Test:** What line number is the broken instruction on? Is the working instruction earlier in the file?

### 2. Token Budget Exhaustion

The skill's total content (SKILL.md + all referenced files) is so large that Claude deprioritizes later or less prominent instructions regardless of position. Distinct from burial — even instructions at line 50 can be dropped if the total volume is high enough.

**Test:** How large is the skill in total? Are there many reference files? Is the broken instruction in a section that competes with high-volume reference content?

### 3. Vague Language

The instruction uses unmeasurable terms that Claude can interpret flexibly.

**Red flags:** "properly", "appropriately", "as needed", "when relevant", "if applicable", "consider", "try to", "good", "clear", "comprehensive"

**Test:** Could two different interpretations of this instruction produce different behavior? If yes, it's vague.

### 4. Missing Example

The working behavior has a concrete example (input → action → output). The broken behavior only has a prose instruction with no example.

**Test:** Does the broken instruction have an example? Does the working instruction? If only the working one does, this is likely the cause.

### 5. Naming Inconsistency

The instruction uses one term, but the target file, variable, or concept uses a different term. Claude never connects them.

**Common variants:** "ledger" vs "log" vs "history", "profile" vs "settings" vs "config", "update" vs "save" vs "write", "session" vs "interaction" vs "conversation"

**Test:** Search for all terms related to the broken feature. Are they consistent everywhere?

### 6. Contradicted Instruction

Another instruction elsewhere in the skill conflicts with the broken one. Claude follows the one it encounters first or the one that's more specific.

**Test:** Search for instructions that could override, limit, or conflict with the broken instruction.

### 7. Conditional Skip

The instruction is gated by a condition that's easy to fail or is evaluated subjectively by Claude.

**Red flags:** "if the session was productive", "if significant progress was made", "when appropriate", "if there are meaningful changes"

**Test:** Is the broken instruction conditional? Is the working instruction unconditional?

### 8. Outside Main Workflow

The skill has a numbered step-by-step workflow, but the broken instruction is mentioned in passing outside that workflow (in a notes section, appendix, or preamble).

**Test:** Is the broken instruction an explicit numbered step in the main workflow? Or is it mentioned elsewhere?

### 9. No Format Specification

The instruction says to update something but doesn't specify the format. Claude either skips it (because it doesn't know how) or writes something that doesn't match what's expected.

**Test:** Does the instruction specify the exact format, structure, or schema of the output?

### 10. Implicit vs Explicit

The working instruction says "you MUST do X". The broken instruction says "the skill also does Y" or "Y should be updated" — passive voice, no direct command.

**Test:** Is the broken instruction a direct imperative ("Update the ledger") or a passive description ("The ledger is updated")?

### 11. Missing Trigger

The working behavior is triggered by an explicit event in the workflow ("after completing step 3, do X"). The broken behavior has no clear trigger — it's stated as something that "should happen" without specifying when.

**Test:** When exactly in the workflow should the broken behavior occur? Is that moment specified?

### 12. File Path Error

The instruction references a file that doesn't exist, has a different name, or is in a different directory than specified.

**Test:** Does the referenced file actually exist at the specified path? Check case sensitivity.

### 13. Capability Mismatch

The instruction asks Claude to do something it cannot do in the current environment (e.g., persist data across sessions without storage, access external APIs without network).

**Test:** Can Claude actually execute this instruction given its current capabilities and environment?

## Phase 4: Determine Root Cause

Classify the root cause as one of:

- **MISSING**: The instruction doesn't exist at all
- **BURIED**: The instruction exists but is too deep in the file (approximately line 300+) to be reliably followed
- **TOKEN_EXHAUSTION**: The skill's total volume causes Claude to deprioritize the instruction regardless of position
- **VAGUE**: The instruction exists but is ambiguous
- **CONTRADICTED**: Another instruction conflicts with it
- **CONDITIONAL_SKIP**: A condition allows Claude to skip it
- **NAMING_MISMATCH**: Inconsistent terminology breaks the connection
- **NO_EXAMPLE**: Working behavior has examples, broken behavior doesn't
- **ORDERING**: Depends on a prior step that may not have completed
- **SCOPE**: Instruction is outside the main workflow and gets deprioritized

## Output

```
## Root Cause
[One sentence. Classification tag + what specifically causes the bug.]

## Evidence
[File names, line numbers, exact text. If MISSING, state where it should be.]

## Working Path
- Trigger: [what initiates it]
- Location: [file:lines]
- Why it works: [brief]

## Broken Path
- Trigger: [what initiates it, or "MISSING"]
- Location: [file:lines, or "NOT FOUND"]
- Failure point: [where and why it breaks]

## Fix
[Minimal change. Show exact text to add, modify, or remove, and where.
Do not refactor. Do not optimize. Do not improve anything unrelated.]

## Verification
[How to confirm the fix. What should the user see after applying it?]

## Related Risks
[Other instructions that share the same failure pattern. Max 3.]
```

## Constraints

- **Fix the bug. Nothing else.** Do not refactor, optimize, or suggest improvements unrelated to the reported bug.
- **Trace, don't guess.** Every claim must reference specific files and lines.
- **Minimal fix.** The smallest change that resolves the bug.
- **Compare what works to what doesn't.** The diff between those paths is where the bug lives.
