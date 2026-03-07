# Trigger Tests

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Direct cover letter request
- **Query:** "write a cover letter for this position" (pastes job description)
- **Expected:** cover-letter activates

### 2. Post-resume cover letter
- **Query:** "now write a cover letter for this role"
- **Expected:** cover-letter activates

### 3. Draft request
- **Query:** "draft a cover letter for Goldman Sachs"
- **Expected:** cover-letter activates

## Should NOT Activate `MANUAL`

### 4. Resume tailoring request
- **Query:** "tailor my resume for this JD"
- **Expected:** resume-analyzer activates, NOT cover-letter

### 5. General writing help
- **Query:** "help me write an email to a recruiter"
- **Expected:** Does NOT activate
