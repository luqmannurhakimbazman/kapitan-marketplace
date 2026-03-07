# Trigger Tests

**Test types:** `MANUAL` -- requires a live Claude Code session.

## Should Activate `MANUAL`

### 1. Direct tailoring request
- **Query:** "tailor my resume for this JD" (pastes job description)
- **Expected:** resume-analyzer activates, reads master resume and candidate-context.md

### 2. Company-specific request
- **Query:** "help me customize my resume for a data engineer role at Stripe"
- **Expected:** resume-analyzer activates, asks for JD if not provided

### 3. ATS optimization request
- **Query:** "optimize my resume for ATS"
- **Expected:** resume-analyzer activates, begins keyword extraction

### 4. Resume with URL JD
- **Query:** "tailor my resume for this role: https://jobs.lever.co/company/12345"
- **Expected:** resume-analyzer activates, fetches JD from URL

### 5. Refactor phrasing
- **Query:** "refactor resume for ML engineer at DeepMind"
- **Expected:** resume-analyzer activates

## Should NOT Activate `MANUAL`

### 6. General resume advice
- **Query:** "what's a good resume format for software engineers?"
- **Expected:** Does NOT activate

### 7. Cover letter only
- **Query:** "write a cover letter for this position"
- **Expected:** cover-letter skill activates, NOT resume-analyzer

### 8. Resume generation after analysis
- **Query:** "generate the resume.tex from these notes"
- **Expected:** resume-tailor activates, NOT resume-analyzer
