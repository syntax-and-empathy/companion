# Jekyll Minimal Mistakes Customization Guide

Complete guide to customizing the Syntax & Empathy Jekyll site built with Minimal Mistakes theme.

## Table of Contents

- [Theme Skins](#theme-skins)
- [Author Profile](#author-profile)
- [Site Footer](#site-footer)
- [Page Layouts](#page-layouts)
- [Per-Page Customization](#per-page-customization)
- [Navigation](#navigation)
- [Analytics & Comments](#analytics--comments)
- [Categories & Tags](#categories--tags)
- [SEO Enhancements](#seo-enhancements)
- [CSS/SCSS Customization](#cssscss-customization)
- [Custom JavaScript](#custom-javascript)
- [Advanced Customization](#advanced-customization)

---

## Theme Skins

Change the overall color scheme by editing `_config.yml:12`:

```yaml
minimal_mistakes_skin: "dark"  # Current selection
```

### Available Skins

| Skin | Description |
|------|-------------|
| `default` | White background, classic look |
| `air` | Light blue-gray, airy feel |
| `aqua` | Blue-green, ocean-inspired |
| `contrast` | High contrast dark mode |
| `dark` | **Current** - Dark background with light text |
| `dirt` | Brown/tan, earthy tones |
| `neon` | Dark with neon color accents |
| `mint` | Light green, fresh look |
| `plum` | Purple theme |
| `sunrise` | Orange/warm, vibrant |

**To preview:** [Minimal Mistakes Skin Gallery](https://mmistakes.github.io/minimal-mistakes/docs/configuration/#skin)

---

## Author Profile

Located in `_config.yml:15-22`. Currently minimal - expand with:

```yaml
author:
  name: "Your Name"
  avatar: "/assets/images/bio-photo.jpg"  # 200x200px recommended
  bio: "Short bio text (2-3 sentences)"
  location: "City, State/Country"
  email: "your.email@example.com"
  links:
    - label: "Website"
      icon: "fas fa-fw fa-link"
      url: "https://www.syntaxandempathy.ai"
    - label: "Email"
      icon: "fas fa-fw fa-envelope-square"
      url: "mailto:your.email@example.com"
    - label: "Twitter"
      icon: "fab fa-fw fa-twitter-square"
      url: "https://twitter.com/username"
    - label: "GitHub"
      icon: "fab fa-fw fa-github"
      url: "https://github.com/username"
    - label: "LinkedIn"
      icon: "fab fa-fw fa-linkedin"
      url: "https://linkedin.com/in/username"
    - label: "Instagram"
      icon: "fab fa-fw fa-instagram"
      url: "https://instagram.com/username"
    - label: "YouTube"
      icon: "fab fa-fw fa-youtube"
      url: "https://youtube.com/@username"
```

### Font Awesome Icons

Use [Font Awesome 5](https://fontawesome.com/v5/search) icons with these prefixes:
- `fas` - Solid icons
- `fab` - Brand icons
- `far` - Regular icons

---

## Site Footer

Add footer links and copyright in `_config.yml`:

```yaml
footer:
  links:
    - label: "Twitter"
      icon: "fab fa-fw fa-twitter-square"
      url: "https://twitter.com/username"
    - label: "GitHub"
      icon: "fab fa-fw fa-github"
      url: "https://github.com/syntax-and-empathy/companion"
    - label: "RSS Feed"
      icon: "fas fa-fw fa-rss-square"
      url: "/feed.xml"

# Custom footer text (HTML allowed)
atom_feed:
  hide: false  # Set to true to hide RSS link in footer
```

---

## Page Layouts

Minimal Mistakes provides several layouts. Currently using `single` and `home`.

### Available Layouts

#### `single` (Default)
Standard page with optional sidebar, TOC, and author profile.

```yaml
---
layout: single
title: "Page Title"
---
```

#### `splash`
Full-width landing page with hero images and CTAs.

```yaml
---
layout: splash
header:
  overlay_color: "#000"
  overlay_filter: "0.5"  # Darken background image
  overlay_image: /assets/images/header.jpg
  actions:
    - label: "Get Started"
      url: "/getting-started/"
      btn_class: "btn--primary"
    - label: "Learn More"
      url: "/about/"
      btn_class: "btn--inverse"
excerpt: "Tagline or excerpt text"
---
```

#### `home`
Blog-style homepage with recent posts (currently set for `index.md`).

```yaml
---
layout: home
author_profile: true
---
```

#### `archive`
Grid or list view of posts/pages.

```yaml
---
layout: archive
title: "All Posts"
permalink: /posts/
---
```

#### `categories` / `tags`
Organize posts by categories or tags.

```yaml
---
layout: categories
permalink: /categories/
---
```

#### `posts`
Posts organized by year/month/day.

```yaml
---
layout: posts
title: "Blog Posts"
permalink: /blog/
---
```

---

## Per-Page Customization

Add these options to any page's front matter:

### Basic Options

```yaml
---
title: "Page Title"
excerpt: "Short description for SEO and previews"
permalink: /custom-url/
classes: wide  # Remove sidebar, use full width
author_profile: false  # Hide author sidebar
sidebar:
  nav: "custom-nav"  # Use custom navigation (define in _data/navigation.yml)
---
```

### Table of Contents

```yaml
---
toc: true
toc_label: "On This Page"  # Custom TOC title
toc_icon: "cog"  # Font Awesome icon name
toc_sticky: true  # Stick TOC to top when scrolling
---
```

### Header Images & Overlays

```yaml
---
header:
  image: /assets/images/header.jpg  # Full-width header image
  teaser: /assets/images/teaser.jpg  # Thumbnail for archives/grids (400x250px)
  og_image: /assets/images/og.jpg  # Social media share image
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
---
```

**Or with overlay:**

```yaml
---
header:
  overlay_image: /assets/images/header.jpg
  overlay_filter: 0.5  # 0 (transparent) to 1 (opaque)
  overlay_color: "#333"  # Solid color overlay
  caption: "Photo caption"
  actions:
    - label: "Call to Action"
      url: "/path/"
---
```

### Feature Rows (Splash Layout)

```yaml
---
layout: splash
feature_row:
  - image_path: /assets/images/feature1.jpg
    alt: "Feature 1"
    title: "Feature 1 Title"
    excerpt: "Description of feature 1"
    url: "/feature1/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - image_path: /assets/images/feature2.jpg
    alt: "Feature 2"
    title: "Feature 2 Title"
    excerpt: "Description of feature 2"
    url: "/feature2/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - image_path: /assets/images/feature3.jpg
    alt: "Feature 3"
    title: "Feature 3 Title"
    excerpt: "Description of feature 3"
    url: "/feature3/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
---

Content here...

{% include feature_row %}
```

### Gallery

```yaml
---
gallery:
  - url: /assets/images/photo1.jpg
    image_path: /assets/images/photo1-thumb.jpg
    alt: "Photo 1"
    title: "Photo 1 caption"
  - url: /assets/images/photo2.jpg
    image_path: /assets/images/photo2-thumb.jpg
    alt: "Photo 2"
    title: "Photo 2 caption"
---

{% include gallery caption="Gallery caption" %}
```

### Read Time & Dates

```yaml
---
read_time: true  # Show estimated read time
show_date: true  # Show publish date
date: 2025-11-13
last_modified_at: 2025-11-13  # Show "Updated" date
---
```

### Social Sharing

```yaml
---
share: true  # Show social share buttons
---
```

### Comments

```yaml
---
comments: true  # Enable comments (requires provider setup in _config.yml)
---
```

---

## Navigation

Edit `_data/navigation.yml` to customize site navigation.

### Simple Navigation (Current)

```yaml
main:
  - title: "Articles"
    url: /articles/
  - title: "Archive"
    url: /archive/
  - title: "Templates"
    url: /templates/
  - title: "About"
    url: /about/
```

### Dropdown Navigation

```yaml
main:
  - title: "Content"
    url: /content/
    children:
      - title: "Articles"
        url: /articles/
      - title: "Archive"
        url: /archive/
      - title: "Templates"
        url: /templates/
  - title: "About"
    url: /about/
```

### Custom Sidebar Navigation

Create named navigation sections for sidebars:

```yaml
docs:
  - title: "Getting Started"
    children:
      - title: "Installation"
        url: /docs/installation/
      - title: "Configuration"
        url: /docs/configuration/
  - title: "Advanced"
    children:
      - title: "Customization"
        url: /docs/customization/
      - title: "Deployment"
        url: /docs/deployment/
```

Use in page front matter:

```yaml
---
sidebar:
  nav: "docs"
---
```

---

## Analytics & Comments

### Google Analytics

Edit `_config.yml`:

```yaml
analytics:
  provider: "google-gtag"  # Recommended (GA4)
  google:
    tracking_id: "G-XXXXXXXXXX"  # Your GA4 measurement ID
    anonymize_ip: false  # true if required by GDPR
```

**Other providers:**
- `google` - Universal Analytics (legacy)
- `google-universal` - Universal Analytics
- `plausible` - Privacy-focused
- `custom` - Custom analytics code

### Plausible Analytics (Privacy-Focused)

```yaml
analytics:
  provider: "plausible"
  plausible:
    domain: "yourdomain.com"
```

### Comments

#### Disqus

```yaml
comments:
  provider: "disqus"
  disqus:
    shortname: "your-disqus-shortname"
```

#### Utterances (GitHub Issues)

```yaml
comments:
  provider: "utterances"
  utterances:
    theme: "github-dark"  # or "github-light"
    issue_term: "pathname"
    repo: "username/repo"
```

#### Giscus (GitHub Discussions)

```yaml
comments:
  provider: "giscus"
  giscus:
    repo_id: "your-repo-id"
    category_name: "Announcements"
    category_id: "your-category-id"
    discussion_term: "pathname"
    reactions_enabled: "1"
    theme: "dark"  # or "light"
```

Enable per page:

```yaml
---
comments: true
---
```

---

## Categories & Tags

### In Posts

Create a post in `_posts/YYYY-MM-DD-title.md`:

```yaml
---
title: "Post Title"
categories:
  - Design
  - AI
tags:
  - prompt-engineering
  - ux
  - frameworks
  - claude
---

Post content...
```

### Create Archive Pages

**Categories page** (`_pages/category-archive.md`):

```yaml
---
title: "Posts by Category"
layout: categories
permalink: /categories/
author_profile: true
---
```

**Tags page** (`_pages/tag-archive.md`):

```yaml
---
title: "Posts by Tag"
layout: tags
permalink: /tags/
author_profile: true
---
```

**Year archive** (`_pages/year-archive.md`):

```yaml
---
title: "Posts by Year"
layout: posts
permalink: /posts/
author_profile: true
---
```

Add to navigation in `_data/navigation.yml`.

---

## SEO Enhancements

### Open Graph & Twitter Cards

Edit `_config.yml`:

```yaml
og_image: /assets/images/site-logo.png  # Default social share image (1200x630px)
twitter:
  username: "yourusername"  # Without @
  card: summary_large_image  # or "summary"

# JSON-LD structured data
social:
  type: Person  # or Organization
  name: Your Name
  links:
    - "https://twitter.com/username"
    - "https://github.com/username"
    - "https://linkedin.com/in/username"
```

### Per-Page SEO

```yaml
---
title: "Page Title"
excerpt: "Meta description (150-160 characters)"
header:
  og_image: /assets/images/custom-og.jpg  # Override default
---
```

---

## CSS/SCSS Customization

### Method 1: Simple Custom CSS

Create `/assets/css/main.scss`:

```scss
---
# Only the main Sass file needs front matter (the dashes are enough)
---

@charset "utf-8";

@import "minimal-mistakes/skins/{{ site.minimal_mistakes_skin | default: 'default' }}"; // skin
@import "minimal-mistakes"; // main partials

/* Custom CSS below this line */

/* Override link colors */
a {
  color: #00adb5;

  &:hover {
    color: #00d9e6;
  }
}

/* Custom font family */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Adjust masthead (header) */
.masthead {
  background-color: #1a1a1a;
  border-bottom: 2px solid #00adb5;
}

/* Custom button styles */
.btn--primary {
  background-color: #00adb5;

  &:hover {
    background-color: #00d9e6;
  }
}

/* Adjust page width */
.page {
  max-width: 1280px;
}

/* Code blocks */
code {
  background-color: #2d2d2d;
  color: #f8f8f2;
}
```

### Method 2: SCSS Variables Override

Create `/assets/css/main.scss` with variable overrides **before** imports:

```scss
---
---

@charset "utf-8";

/* ==========================================================================
   Variables Override
   ========================================================================== */

/* Typography */
$sans-serif: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !default;
$monospace: Monaco, Consolas, "Courier New", monospace !default;

$type-size-1: 2.441em !default;  // ~39.056px - h1
$type-size-2: 1.953em !default;  // ~31.248px - h2
$type-size-3: 1.563em !default;  // ~25.008px - h3
$type-size-4: 1.25em !default;   // ~20px - h4
$type-size-5: 1em !default;      // ~16px - h5
$type-size-6: 0.75em !default;   // ~12px - h6

/* Colors */
$background-color: #252a34 !default;
$text-color: #eaeaea !default;
$primary-color: #00adb5 !default;
$border-color: mix(#fff, $background-color, 20%) !default;
$code-background-color: mix(#000, $background-color, 15%) !default;
$code-background-color-dark: mix(#000, $background-color, 20%) !default;
$link-color: $primary-color !default;
$link-color-hover: mix(#fff, $link-color, 25%) !default;
$link-color-visited: mix(#000, $link-color, 25%) !default;
$masthead-link-color: $primary-color !default;
$masthead-link-color-hover: mix(#fff, $masthead-link-color, 25%) !default;

/* Spacing */
$right-sidebar-width-narrow: 200px !default;
$right-sidebar-width: 300px !default;
$right-sidebar-width-wide: 400px !default;

/* Breakpoints */
$small: 600px !default;
$medium: 768px !default;
$medium-wide: 900px !default;
$large: 1024px !default;
$x-large: 1280px !default;
$max-width: $x-large !default;

/* Border radius */
$border-radius: 4px !default;
$box-shadow: 0 1px 1px rgba(0, 0, 0, 0.125) !default;

/* Now import theme */
@import "minimal-mistakes/skins/{{ site.minimal_mistakes_skin | default: 'default' }}";
@import "minimal-mistakes";

/* Custom overrides after imports */
```

### Method 3: Custom Skin

Create a custom skin in `_sass/minimal-mistakes/skins/_custom.scss`:

```scss
/* ==========================================================================
   Custom skin
   ========================================================================== */

/* Colors */
$background-color: #1a1a2e !default;
$text-color: #eaeaea !default;
$primary-color: #0f3460 !default;
$border-color: #16213e !default;
$code-background-color: #0f3460 !default;
$footer-background-color: #16213e !default;
$link-color: #e94560 !default;
$masthead-link-color: #eaeaea !default;
$navicon-link-color-hover: mix(#fff, $primary-color, 75%) !default;
$form-background-color: #0f3460 !default;
$muted-text-color: mix(#fff, $text-color, 20%) !default;

/* Syntax highlighting (optional) */
.highlight {
  background-color: $code-background-color;
}
```

Then in `_config.yml`:

```yaml
minimal_mistakes_skin: "custom"
```

### Common CSS Customizations

#### Adjust Content Width

```scss
.page {
  @include breakpoint($large) {
    max-width: 1280px;  // Default is 1024px
  }
}
```

#### Custom Fonts

```scss
/* In <head> or _includes/head/custom.html */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-weight: 700;
}
```

#### Custom Masthead (Header)

```scss
.masthead {
  background-color: #1a1a1a;
  border-bottom: 2px solid $primary-color;

  .site-title {
    font-size: 1.5em;
    font-weight: 700;
  }

  .greedy-nav {
    a {
      color: $masthead-link-color;

      &:hover {
        color: $primary-color;
      }
    }
  }
}
```

#### Custom Footer

```scss
.page__footer {
  background-color: #1a1a1a;
  color: #888;

  a {
    color: $primary-color;
  }

  footer {
    padding: 2em 0;
  }
}
```

#### Rounded Images

```scss
img {
  border-radius: 8px;
}

.author__avatar img {
  border-radius: 50%;  // Circular avatar
}
```

#### Custom Buttons

```scss
.btn {
  &--custom {
    background-color: #00adb5;
    color: #fff;

    &:hover {
      background-color: #00d9e6;
    }
  }
}
```

#### Syntax Highlighting Theme

Override code block colors:

```scss
/* Based on Monokai */
.highlight {
  background: #272822;
  color: #f8f8f2;

  .c { color: #75715e; } /* Comment */
  .k { color: #f92672; } /* Keyword */
  .s { color: #e6db74; } /* String */
  .n { color: #f8f8f2; } /* Name */
  .o { color: #f92672; } /* Operator */
  .p { color: #f8f8f2; } /* Punctuation */
}
```

---

## Custom JavaScript

### Method 1: Site-wide Custom Scripts

Create `_includes/head/custom.html`:

```html
<!-- Custom JavaScript -->
<script>
  // Your custom JS here
  console.log('Custom JS loaded');

  // Example: Add smooth scrolling
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    });
  });
</script>
```

### Method 2: External Script File

Create `/assets/js/custom.js`:

```javascript
// Custom JavaScript
(function() {
  'use strict';

  // Your code here
  console.log('Custom JS loaded from external file');

  // Example: Dark mode toggle
  const toggleDarkMode = () => {
    document.body.classList.toggle('dark-mode');
  };

})();
```

Include in `_includes/head/custom.html`:

```html
<script src="{{ '/assets/js/custom.js' | relative_url }}"></script>
```

### Method 3: Per-Page Scripts

In page front matter:

```yaml
---
title: "Page with Custom JS"
---

<script>
  // Page-specific JavaScript
  document.addEventListener('DOMContentLoaded', function() {
    console.log('Page-specific JS loaded');
  });
</script>
```

---

## Advanced Customization

### Custom Includes

Override theme includes by creating files in `_includes/` directory:

- `_includes/head/custom.html` - Custom `<head>` content
- `_includes/footer/custom.html` - Custom footer content
- `_includes/analytics-providers/custom.html` - Custom analytics
- `_includes/comments-providers/custom.html` - Custom comments

### Custom Layouts

Create custom layouts in `_layouts/` directory:

```liquid
---
layout: default
---

<div class="custom-layout">
  <article class="page" itemscope itemtype="https://schema.org/CreativeWork">
    <div class="page__inner-wrap">
      <header>
        <h1 class="page__title">{{ page.title }}</h1>
      </header>
      <section class="page__content">
        {{ content }}
      </section>
    </div>
  </article>
</div>
```

### Collections

Create custom content collections in `_config.yml`:

```yaml
collections:
  portfolio:
    output: true
    permalink: /:collection/:path/
  recipes:
    output: true
    permalink: /:collection/:path/

defaults:
  - scope:
      path: ""
      type: portfolio
    values:
      layout: single
      author_profile: false
      share: true
```

### Localization

Support multiple languages:

```yaml
locale: "en-US"  # or "es-ES", "fr-FR", etc.

# Custom text strings
words_per_minute: 200
read_more: "Read more"
related_label: "You may also enjoy"
```

---

## File Locations Reference

```
companion/
├── _config.yml                     # Main configuration
├── _data/
│   └── navigation.yml              # Navigation menus
├── _includes/
│   ├── head/custom.html           # Custom <head> content
│   └── footer/custom.html         # Custom footer content
├── _layouts/                       # Custom layouts (optional)
├── _pages/                         # Site pages
├── _posts/                         # Blog posts (YYYY-MM-DD-title.md)
├── _sass/
│   └── minimal-mistakes/
│       └── skins/
│           └── _custom.scss       # Custom skin (optional)
├── assets/
│   ├── css/
│   │   └── main.scss              # Custom CSS/SCSS
│   ├── js/
│   │   └── custom.js              # Custom JavaScript
│   └── images/                    # Images
└── index.md                        # Homepage
```

---

## Quick Reference: Most Common Changes

### 1. Change Theme Color
Edit `_config.yml`: `minimal_mistakes_skin: "dark"`

### 2. Add Profile Photo
1. Add image to `/assets/images/bio-photo.jpg`
2. Edit `_config.yml`: `avatar: "/assets/images/bio-photo.jpg"`

### 3. Customize Colors
Create `/assets/css/main.scss` with color overrides

### 4. Add Google Analytics
Edit `_config.yml`: Add `analytics` section with tracking ID

### 5. Enable Comments
Edit `_config.yml`: Add `comments` section with provider details

### 6. Add Social Links
Edit `_config.yml`: Expand `author.links` array

### 7. Customize Navigation
Edit `_data/navigation.yml`: Modify `main` array

### 8. Change Homepage Layout
Edit `index.md`: Change `layout: home` to `layout: splash`

---

## Resources

- [Minimal Mistakes Documentation](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)
- [Configuration Options](https://mmistakes.github.io/minimal-mistakes/docs/configuration/)
- [Layouts](https://mmistakes.github.io/minimal-mistakes/docs/layouts/)
- [Font Awesome Icons](https://fontawesome.com/v5/search)
- [Liquid Syntax](https://shopify.github.io/liquid/)
- [Kramdown Syntax](https://kramdown.gettalong.org/syntax.html)

---

**Last Updated:** 2025-11-13
