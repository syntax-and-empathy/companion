# Repository Data & Navigation

This directory stores the structured data files that define the "Syntax & Empathy Companion" repository's core functionality, primarily its site-wide navigation.

---

## Wayfinding

**You are here:** root > **_data**

**Up:** [../](../) - Return to the main repository
**Home:** [/](/) - Return to repository root

---

## TL;DR

**In 3 sentences or less:**
This `_data` directory houses essential structured data files, most notably `navigation.yml`, which dictates the entire repository's menu and wayfinding. It's crucial for maintaining a consistent and intuitive user experience across the "Syntax & Empathy Companion" content. Understanding its contents is key for any repository maintainer or contributor looking to update site structure or navigation.

**Key takeaway:** This directory is the single source of truth for the repository's global navigation structure.

**Time investment:** 2 min | Directory Guide

---

## At a Glance

| | |
|---|---|
| **What** | Structured data files for site configuration and navigation. |
| **Why** | Defines the repository's organizational structure and menu navigation. |
| **Who** | Repository maintainers, content creators, and developers. |
| **When** | Ongoing; updated as content and sections are added or modified. |
| **Status** | Published |

**Quick Start:** [`navigation.yml`](navigation.yml) - to understand the site's menu structure.

---

## Overview

The `_data` directory is a foundational component of the "Syntax & Empathy Companion" repository, serving as the central hub for structured data files. Its primary content, `navigation.yml`, is vital for defining the main menu and sub-navigation links, ensuring a logical and consistent user experience. This directory is essential for repository maintainers to understand for effective content organization, site architecture, and future scalability.

<details>
<summary><strong>About This Content</strong> (click to expand)</summary>

### Purpose & Context

The purpose of this directory is to centralize configuration and data files that drive dynamic aspects of the repository, separate from markdown content. Specifically, `navigation.yml` defines the hierarchical structure of the site's menu, mapping content sections to user-friendly links. This ensures that the repository's various topics—design leadership, AI collaboration, technical workflows—are easily discoverable and navigable for design professionals.

### Background

In static site generator environments (such as Jekyll, which commonly uses `_data` directories), this folder is a standard convention for storing global data. This data can then be programmatically accessed by templates to dynamically generate elements like navigation menus, lists, or other structured content, promoting maintainability and consistency.

</details>

---

## Key Topics

### Core Concepts

-   **Site Navigation** - Defines the menu structure for the entire repository.
-   **Structured Data** - Contains data in a machine-readable format (YAML).
-   **Repository Configuration** - Core data that influences how the site functions.

<details>
<summary><strong>Detailed Topic Breakdown</strong> (click to expand)</summary>

### Site Navigation
The [`navigation.yml`](navigation.yml) file within this directory is the single source of truth for the repository's primary and secondary menus. It specifies the order, labels, and target URLs for all top-level sections and their sub-pages, ensuring a consistent user journey.

### Structured Data
Files here, primarily in YAML format, provide data in a structured, hierarchical manner. This allows for easy parsing and use by the site's templating engine, making content management more efficient and less prone to errors compared to hardcoding links.

### Repository Configuration
While `_config.yml` handles overall site settings, files in `_data` often manage specific, large-scale datasets that influence the site's presentation or functionality, such as navigation, author lists, or taxonomies, enabling flexible site architecture.

</details>

---

## Key Takeaways

**You'll learn:**
1.  How the repository's main navigation is structured and controlled.
2.  The role of structured data files in maintaining site consistency.
3.  Where to modify menu links and section ordering for new content.

**You'll be able to:**
-   Identify the file responsible for main site navigation.
-   Understand the YAML format used for data storage.
-   Contribute to or update the repository's navigation scheme.

---

## What's Inside

### Start Here

**[`navigation.yml`](navigation.yml)** - This file defines the primary and secondary navigation menus for the entire repository. Start here to understand the site's structural blueprint.

### Supporting Materials

<details>
<summary><strong>View All Files & Directories</strong> (click to expand)</summary>

```
_data/
└── navigation.yml    Defines the repository's primary navigation menu
```

#### Directory Guide

-   **`navigation.yml`** - This YAML file contains an array of navigation items, each specifying a title, URL, and potentially nested sub-items, forming the backbone of the site's menu system.

</details>

---

## How to Navigate

### Recommended Path

**For the complete experience:**
1.  Start with [`navigation.yml`](navigation.yml) to see the current menu structure.
2.  Understand the YAML syntax for defining links and nested items.
3.  Refer to the live "Syntax & Empathy Companion" site to see how this data translates into the user interface.

### Alternative Paths

**If you're short on time:** Quickly review the top-level items in [`navigation.yml`](navigation.yml) to grasp the main sections.
**If you're looking for specific information:** Use your text editor's search function within [`navigation.yml`](navigation.yml) for specific menu labels or URLs.
**If you want visual context first:** Browse the live "Syntax & Empathy Companion" site and then examine [`navigation.yml`](navigation.yml) to see its corresponding data structure.

**Tip:** Changes to [`navigation.yml`](navigation.yml) will directly impact the site's global menu, so always test thoroughly in a development environment before deploying.

---

## Prerequisites & Context

<details>
<summary><strong>What to know before reading</strong> (click to expand)</summary>

### Helpful Background

-   Basic understanding of YAML syntax.
-   Familiarity with static site generator conventions (e.g., Jekyll's `_data` directory).
-   Knowledge of the repository's overall content structure and hierarchy.

### Related Reading

If you're new to this topic, you might want to start with:
-   [YAML Documentation](https://yaml.org/spec/1.2/spec.html) - For understanding the data format.
-   [Jekyll Data Files Documentation](https://jekyllrb.com/docs/datafiles/) - For context on `_data` directory usage.

</details>

---

## Related Content

### Within This Repository

**Related articles:**
-   [`_config.yml`](../_config.yml) - Site-wide configuration file, which often interacts with data files.
-   [`index.md`](../index.md) - The repository's homepage, which utilizes this navigation.

### Part of a Series

This directory is an integral part of the repository's core infrastructure.

---

## What's Next

**After reading this, you might want to:**
1.  Explore the main content directories (e.g., `design-leadership/`, `ai-collaboration/`) to see how they align with the navigation defined here.
2.  Consider proposing updates to the navigation if new content is added or sections are reorganized.
3.  Learn more about Jekyll's data files feature for advanced customization of the repository.

**Apply what you learned:**
-   Draft a new navigation item for a hypothetical new content section.
-   Verify the current navigation against the live site for consistency.

---

## References & Citations

<details>
<summary><strong>Sources & Further Reading</strong> (click to expand)</summary>

### Recommended Reading

-   YAML Ain't Markup Language (YAML) Official Website: [https://yaml.org/](https://yaml.org/)
-   Jekyll Data Files Documentation: [https://jekyllrb.com/docs/datafiles/](https://jekyllrb.com/docs/datafiles/)

</details>

---

## Metadata

<details>
<summary><strong>Content Information</strong> (click to expand)</summary>

| | |
|---|---|
| **Created** | 2023-10-27 |
| **Last Updated** | 2023-10-27 |
| **Version** | 1.0 |
| **Status** | Published |
| **Content Type** | Directory Guide |
| **Reading Time** | 2 min |
| **Word Count** | ~550 |
| **Author** | Syntax & Empathy Companion Maintainers |
| **Tags** | data, navigation, yaml, configuration, structure, directory, Jekyll |

</details>

---

## Engage

**Found this helpful?** Star this repository or share your feedback in an issue.
**Have questions?** Open an issue on GitHub to discuss the repository's structure.
**Spotted an issue?** Please report it via a GitHub issue.

---

**Stay Updated:** Watch this repository on GitHub for updates.