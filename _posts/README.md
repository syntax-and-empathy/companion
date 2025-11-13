# Posts Directory

This directory contains blog posts for the Chirpy theme homepage and archives.

## Purpose

The `_posts` directory is where Jekyll/Chirpy looks for blog posts to display on:
- Homepage (latest posts)
- Archive pages (by date, category, tag)
- RSS feed

## File Naming Convention

Posts must follow Jekyll's strict naming convention:

```
YYYY-MM-DD-title-with-hyphens.md
```

**Examples:**
- `2025-11-13-setup-jekyll-chirpy-theme.md`
- `2025-10-12-git-github-setup-guide.md`

## Front Matter Template

Each post requires YAML front matter at the top:

```yaml
---
layout: post
title: "Your Post Title"
date: 2025-11-13 10:00:00 -0500
categories: [Category1, Category2]
tags: [tag1, tag2, tag3]
author: <author_id>
toc: true
comments: true
---

Your post content starts here...
```

### Required Fields:
- `layout: post` - Must be "post" for blog posts
- `title` - Post title (use quotes if it contains special characters)
- `date` - Publication date and time with timezone

### Optional but Recommended:
- `categories` - Array of categories (appears in archives)
- `tags` - Array of tags (appears in tag cloud/archives)
- `author` - Author identifier (defaults to site author)
- `toc` - Table of contents (true/false)
- `comments` - Enable comments (true/false)
- `image` - Featured image for post
- `pin` - Pin post to top of homepage (true/false)

## Relationship to Articles Collection

The existing `02-articles/` directory contains a separate content collection with its own structure. That content organization is preserved for:
- Long-form comprehensive guides
- Multi-file articles with extensive assets
- Complex source material organization

**Use `_posts/` for:**
- Blog-style updates and announcements
- Shorter focused articles
- Time-sensitive content
- Content you want featured on the homepage

**Use `02-articles/` collection for:**
- Comprehensive guides with extensive research
- Articles with complex asset organization
- Content that doesn't fit blog chronology

## Migration Notes

To make existing articles from `02-articles/` appear on the Chirpy homepage:

1. Create a post file in `_posts/` with proper naming
2. Either:
   - Copy article content into the post file, OR
   - Create a summary with a link to the full article in the collection

## Examples

See the Chirpy documentation for post examples:
- https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Writing-a-New-Post

## Current Status

This directory is currently empty. Add posts here to populate the Chirpy homepage and archives.
