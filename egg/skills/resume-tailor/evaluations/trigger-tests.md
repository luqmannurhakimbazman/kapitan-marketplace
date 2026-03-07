# Trigger Tests

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Direct generation request
- **Query:** "generate the resume now"
- **Expected:** resume-tailor activates, reads notes.md

### 2. Post-analysis continuation
- **Query:** "build the resume from these notes"
- **Expected:** resume-tailor activates

### 3. Explicit tailor request
- **Query:** "create resume.tex"
- **Expected:** resume-tailor activates

## Should NOT Activate `MANUAL`

### 4. Analysis request
- **Query:** "analyze this JD for my resume"
- **Expected:** resume-analyzer activates, NOT resume-tailor

### 5. Cover letter request
- **Query:** "write a cover letter"
- **Expected:** cover-letter activates, NOT resume-tailor
