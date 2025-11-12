# README Template Examples - Summary

This document provides 4 example READMEs showing how the templates would be applied at different directory levels in the repository, from deepest article content to repository root.

---

## Example Overview

| Level | Path | Template Used | File |
|-------|------|---------------|------|
| **1** | `02-articles/251109-on-authenticity/1109-not-magic-beyond-measurement/` | Content Directory | [EXAMPLE-1-deep-content.md](./EXAMPLE-1-deep-content.md) |
| **2** | `02-articles/251109-on-authenticity/` | Index Directory | [EXAMPLE-2-mid-index.md](./EXAMPLE-2-mid-index.md) |
| **3** | `02-articles/` | Index Directory | [EXAMPLE-3-articles-directory.md](./EXAMPLE-3-articles-directory.md) |
| **4** | `/` (root) | Index Directory | [EXAMPLE-4-root.md](./EXAMPLE-4-root.md) |

---

## UX Patterns Demonstrated

### 1. Wayfinding 🧭

Each example shows **breadcrumb trails** that become progressively shorter as you move up:

**Level 1 (Deep):** `02-articles > 251109-on-authenticity > 1109-not-magic-beyond-measurement`
**Level 2 (Mid):** `02-articles > 251109-on-authenticity`
**Level 3 (Top):** `02-articles`
**Level 4 (Root):** `Repository Root`

All include:
- "You are here" location indicator
- Up link to parent directory (with description)
- Home link to repository root
- Related directories based on siblings

### 2. Progressive Disclosure 📋

Information reveals progressively based on user interest:

**Always Visible (Above the fold):**
- TL;DR / Quick summary
- At a Glance table with key facts
- Quick Start guidance

**Expandable (Click to reveal):**
- Detailed context and background
- Full directory trees
- Metadata and technical details
- References and citations
- Complete topic breakdowns

**Pattern:** Essential information first, details on demand.

### 3. Multiple Navigation Paths 🗺️

Each level provides different ways to navigate based on user needs:

**Content Template (Example 1):**
- Recommended Path (for complete experience)
- Alternative Paths (for time-constrained or specific needs)
- Prerequisites section (what to know first)
- What's Next (where to go after)

**Index Template (Examples 2-4):**
- Where to Start (conditional based on goals)
- How to Explore (multiple browsing strategies)
- Featured Content (curated entry points)
- Related Directories (lateral navigation)

### 4. Information Scent 🎯

Clear indicators help users predict what they'll find:

- **"Best for"** statements in At a Glance tables
- **Descriptive link text** explaining what's at each destination
- **Content type indicators** (📄 Essay, 🎓 Tutorial, 📖 Case Study, etc.)
- **Reading time estimates** for planning engagement
- **Status badges** (Published, Draft, Active, Archive)

### 5. Scannability 👀

Visual hierarchy and formatting support quick scanning:

- **Tables** for quick reference (At a Glance sections)
- **Emoji landmarks** for visual wayfinding
- **Clear headers** with consistent hierarchy
- **Horizontal rules** separating major sections
- **Bold keywords** highlighting key information
- **Bulleted lists** for easy scanning

---

## Template Adaptation by Level

### Deep Content (Example 1)

**Template:** Content Directory
**Focus:** Helping users understand and engage with specific content
**Key Sections:**
- TL;DR (30-second summary)
- What you'll learn and be able to do
- Recommended reading path with alternatives
- Prerequisites and context
- Apply what you learned section

### Mid-Level Index (Example 2)

**Template:** Index Directory
**Focus:** Navigating a collection of related content
**Key Sections:**
- Collection overview and purpose
- Featured content (where to start)
- Complete directory listing with descriptions
- Where to Start (conditional guidance)
- How to Explore (multiple strategies)

### Top-Level Category (Example 3)

**Template:** Index Directory
**Focus:** Organizing different collections and types
**Key Sections:**
- Category purpose and scope
- Featured collections
- Organization logic explained
- What's Next (connections to other areas)
- Metadata about the category

### Repository Root (Example 4)

**Template:** Index Directory (adapted)
**Focus:** Repository-wide orientation and navigation
**Key Sections:**
- Repository purpose and philosophy
- Complete structure overview
- Multiple navigation paths through content
- About section explaining approach
- Engagement mechanisms

---

## How Information Flows Between Levels

### Breadcrumb Context

Each level provides context for the levels below:

**Root** → Explains overall repository purpose
**02-articles/** → Positions articles within repository context
**251109-on-authenticity/** → Shows how collection fits in articles
**1109-not-magic-beyond-measurement/** → References its place in collection

### Related Content Links

Each level connects to:
- **Parent** (up one level)
- **Home** (back to root)
- **Siblings** (same level, different branches)
- **Children** (deeper levels)
- **Related** (different branches, relevant content)

Example from Level 1:
- Up: Collection overview
- Home: Repository root
- Siblings: Other articles in same collection
- Related: Articles from other collections on similar topics

### Progressive Detail

Information becomes more specific as you go deeper:

**Root:** "Design leadership + AI collaboration"
**02-articles/:** "In-depth explorations and multi-part series"
**251109-on-authenticity/:** "Multi-part exploration of authenticity in AI work"
**1109-not-magic-beyond-measurement/:** "Framework for evaluating beyond metrics"

---

## AI Generation Notes

When the GitHub Action generates these READMEs using Gemini 2.5 Flash, it will:

1. **Collect context** about each directory:
   - Path and breadcrumb trail
   - Parent directory summary
   - Sibling directories
   - Child directories (with their README summaries if available)
   - Actual files present

2. **Select appropriate template:**
   - Content template for directories with content files
   - Index template for directories organizing subdirectories

3. **Apply UX patterns:**
   - Build accurate breadcrumbs from path
   - Create wayfinding links to parent/home
   - Suggest logical starting points
   - Generate multiple navigation strategies
   - Include sibling directories in "Related" sections

4. **Maintain consistency:**
   - Use similar tone and language across levels
   - Reference parent/sibling READMEs for context
   - Follow established naming and organizational patterns

---

## Key Differences from Generic READMEs

These templates go beyond standard documentation by:

1. **Active wayfinding** - Always showing where you are and where you can go
2. **Progressive disclosure** - Respecting user time with scannable summaries
3. **Multiple paths** - Supporting different user goals and constraints
4. **Clear information scent** - Predicting what's at each destination
5. **Visual landmarks** - Using emojis and formatting for quick orientation
6. **Actionable next steps** - Always suggesting where to go next
7. **Context preservation** - Showing how each piece fits into the whole

---

## Files

- **[EXAMPLE-1-deep-content.md](./EXAMPLE-1-deep-content.md)** - Individual article README
- **[EXAMPLE-2-mid-index.md](./EXAMPLE-2-mid-index.md)** - Article collection README
- **[EXAMPLE-3-articles-directory.md](./EXAMPLE-3-articles-directory.md)** - Top-level category README
- **[EXAMPLE-4-root.md](./EXAMPLE-4-root.md)** - Repository root README

These examples show the templates in action and can serve as references when the GitHub Action generates actual READMEs for your repository.
