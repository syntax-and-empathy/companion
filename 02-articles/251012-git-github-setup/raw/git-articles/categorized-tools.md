# DESIGN VERSION CONTROL TOOLS & PLATFORMS

## Application: Design Version Control (Native Built-in Version Control)

### Figma (Built-in Branching)
Figma provides native version control functionality integrated directly into the platform, available on Organization and Enterprise plans. Figma's branching allows designers to create unlimited branches from main files, work independently without disrupting production designs, and merge changes with visual conflict resolution. Unlike traditional Git, Figma branches are auto-saved with real-time multiplayer collaboration, eliminating the need for manual commits.

**Pricing:**
- Professional: $20/month per Full seat (was $15, increased March 2025)
- Organization: $55/month per Full seat (was $45, increased March 2025) - Annual only
- Enterprise: $90/month per Full seat (was $75, increased March 2025) - Annual only
- Dev seats: $15 (Professional), $25 (Organization), $35 (Enterprise)
- Collab seats: $5/month across all plans

**Key features:**
- Unlimited branch creation with automatic version tracking
- Visual side-by-side conflict resolution during merges
- Real-time multiplayer collaboration within branches
- Automatic version history with unlimited retention and named versions
- Branch review and approval workflows before merging

**Status:** Active and continuously evolving. Price increases implemented in March 2025 with new seat structure (Full, Dev, Collab, View seats). Best for teams needing robust design collaboration with built-in version control.

***

### Sketch Cloud
Sketch Cloud provides native version control capabilities built directly into Sketch as part of the standard subscription. While not as robust as Abstract's branching system, Sketch Cloud includes comprehensive version history, starred versions that allow teams to mark important milestones, and granular control over which versions Viewers and Guests can access. The system automatically tracks changes as designers work, eliminating the need for manual commits or version management.

**Pricing:**
- Standard Subscription: $10/month per editor (billed annually) or $12/month (monthly)
- Business Subscription: $22/month per editor (billed annually)
- Mac-only License: $120/seat (one-time, includes 1 year updates)

**Key features:**
- Automatic version history tracking for all cloud documents
- Starred versions for marking important milestones
- Version access controls for team members with different permission levels
- Cloud synchronization across devices and team members
- Integration with Sketch's native collaboration features

**Status:** Active with regular updates. Includes both native Mac app and web app access. Best for designers who prefer Sketch's workflow and need basic version control without complex Git-based branching.

---

## Application: Design Version Control (Non-Native Extensions & Third-Party Solutions)

### Figma Plugins (Version Control Related)
Figma plugins extend Figma's capabilities with Git integration and enhanced versioning features through community-developed plugins.

**Pricing:** Varies by plugin (many free, some premium)

**Notable plugins:**
- **GitHub Plugin**: Connects Dev Mode to GitHub repositories, allowing teams to link files, issues, and pull requests directly to Figma designs and components
- **Design Version**: Creates detailed changelogs with custom change records, release candidates, and automatic Figma version history integration for comprehensive design documentation
- **Tokens Studio**: Syncs design tokens to GitHub, GitLab, or other Git providers with commit messages, enabling designers to push/pull token changes through actual Git workflows
- **Code Connect**: Bridges design components to actual code repositories, linking Figma components to their implementation in GitHub for AI-assisted development

**Status:** Active community-developed plugins that extend Figma's native capabilities. Best for teams needing deeper Git integration or specialized version control workflows.

***

### Abstract (Branches)
Abstract is the most mature Git-like version control solution specifically designed for Sketch, used by one in three Fortune 500 companies. It provides comprehensive branching and merging workflows modeled after Git, with a designer-friendly interface that doesn't require command-line knowledge. Abstract allows teams to work on parallel branches, create child branches for sub-explorations, review changes through visual diffs, and merge approved work back to the main branch.

**Pricing:**
- Pro: $35/month per contributor (billed annually) or $44/month (monthly)
- Enterprise: Custom pricing with unlimited storage and seats

**Key features:**
- True Git-based branching with parent and child branch support
- Visual diff comparisons and review requests with Collections
- Commit-based version history with detailed branch summaries
- Inspect mode for design specs and CSS export
- Integrations with Jira, GitHub, and project management tools

**Status:** Active and mature. Most comprehensive Git-like version control for Sketch. Best for design teams requiring true branching workflows and enterprise-grade version control for Sketch files.

***

### Kactus
Kactus takes a unique approach by integrating actual Git directly into Sketch workflows, transforming Sketch files into a format that Git can understand and version. This open-source tool enables designers to use the full Git workflow including branches, pull requests, commits, and team collaboration through platforms like GitHub, GitLab, or Bitbucket. Kactus allows multiple designers to work on the same file simultaneously and automatically merges non-conflicting changes.

**Pricing:** Free (open-source)

**Key features:**
- True Git integration enabling participation in developer repositories
- Automatic file format conversion for Git compatibility
- Support for full Git workflow including branches and pull requests
- Automatic merging of non-conflicting changes
- Open-source codebase with GitHub repository

**Status:** Active open-source project. Best for teams already using Git who want designers to participate in the same repositories using actual Git workflows.

***

### Sympli Versions
Sympli Versions offers Git-like version control specifically for Sketch, with the unique approach of connecting directly to existing Git platforms like GitHub, Bitbucket, or Azure DevOps rather than using proprietary cloud infrastructure. This allows design files to integrate with developer workflows and existing version control systems. Sympli Versions provides branching capabilities, transparent change history, conflict resolution, and tight macOS Finder integration.

**Pricing:** Part of Sympli platform pricing (contact for pricing)

**Key features:**
- Direct integration with GitHub, Bitbucket, and Azure DevOps
- Branching and merging workflows using actual Git repositories
- Visual conflict resolution and change tracking
- macOS Finder integration for native file management
- Part of broader Sympli platform with design handoff features

**Status:** Active as part of the Sympli design handoff platform. Best for teams wanting to version Sketch files directly in their existing Git infrastructure.

***

### Folio
Folio is a Mac-only application that provides GUI-based version control built directly on top of Git, supporting Photoshop, Illustrator, and Sketch files with visual previews for each saved version. Folio's distinctive "waterfall display" presents all directories and subdirectories visually in the same hierarchy, making navigation simpler than traditional Git interfaces. The tool integrates with existing version control systems like GitHub, GitLab, and BitBucket, allowing designers to participate in actual Git workflows without command-line knowledge.

**Pricing:** Contact for pricing

**Key features:**
- Visual Git interface with preview thumbnails for design files
- Support for multiple Adobe tools and Sketch
- Integration with GitHub, GitLab, and BitBucket
- Waterfall directory display for simplified navigation
- Actual Git repository management with designer-friendly UI

**Status:** Active Mac-only solution. Best for designers working with Adobe tools and Sketch who need visual Git management without command-line expertise.

***

### Plant
Plant is a simple version control tool for Sketch that focuses on ease of use without requiring understanding of Git concepts. It uses designer-friendly language and provides straightforward version management with conflict resolution at the artboard level.

**Pricing:** Discontinued (was freemium)

**Key features:**
- Simple version control without Git branching concepts
- Easy-to-understand language ("send changes", "load changes", "restore version")
- Page-level list of changes to documents
- Artboard-level conflict resolution
- Notifications when teammates make changes

**Status:** Discontinued. Was a simplified alternative to Git-based version control for Sketch users who wanted basic versioning without learning Git concepts.

***

### Pics.io
Pics.io is a digital asset management platform that includes version control capabilities for design files, particularly focused on visual comparison and approval workflows.

**Pricing:** Contact for pricing (enterprise DAM solution)

**Key features:**
- Visual comparison tool for images and videos
- Revision approval and disapproval workflows
- Upload multiple revisions with version tracking
- Centralized project-related communication
- Real-time notifications via email and Slack

**Status:** Active enterprise DAM platform. Best for large organizations managing design assets with approval workflows and version tracking as part of broader digital asset management.

***

### Trunk Version Control
Trunk Version Control is a design-focused version control system that works with Sketch and Adobe Photoshop, offering a freemium model with online access.

**Pricing:** Freemium model

**Key features:**
- Works with Sketch and Adobe Photoshop
- Freemium model with online access
- Focused on designer-specific version control needs

**Status:** Limited information available. Positioned as a designer-friendly version control option for Sketch and Photoshop users.

***

### Canva (Design Tools)
Canva is a cloud-based design platform offering limited version control capabilities as part of its broader design creation toolset.

**Pricing:** 
- Free: $0
- Pro: $15/month per person (billed annually)
- Teams: $30/month for first 5 people (billed annually)
- Enterprise: Custom pricing

**Key features:**
- Cloud-based design creation
- Version history with restore capability
- Team collaboration features
- Template-based design system
- Brand kit for consistency

**Status:** Active with extensive user base. Version control capabilities are basic compared to specialized tools. Best for teams needing simple design creation with basic version history rather than complex design system management.

---

# AI NO-CODE & LOW-CODE DEVELOPMENT PLATFORMS

## Application: Full-Stack Development (Native AI Integration)

### Bolt.new
Bolt.new is an AI-native browser-based coding environment designed for instant full-stack application development. Built by StackBlitz, Bolt enables users to generate production-ready web applications through natural language prompts, with real-time collaboration similar to multiplayer Google Docs for coding. The platform supports leading AI models including Claude and GPT-4, allowing users to build complete apps with frontend, backend, and database integration directly in the browser without local setup.

**Pricing:** 
- Free: $0 (limited AI generations, smaller projects)
- Pro: Token-based pricing starting at $25/month (10M tokens), up to $2,000/month (1.2B tokens)
  - $25/month: 10M tokens
  - $50/month: 26M tokens
  - $100/month: 55M tokens
  - $200/month: 120M tokens
  - Higher tiers available up to $2,000/month
- Teams: Pro pricing + $5/member
- Enterprise: Custom pricing

**Key features:**
- AI-powered full-stack app generation from text prompts
- Real-time collaboration with team members
- Support for modern frameworks (React, Vue, Next.js, Node.js)
- In-browser development environment with no setup required
- Direct deployment capabilities with custom domains
- Token-based usage model with rollover starting July 2025

**Status:** Active and rapidly growing. Token rollover begins July 2025 (tokens valid for 2 months total). Best for solo developers, indie hackers, and fast-moving teams building MVPs or production apps. Main limitation is aggressive token consumption—users report burning through millions of tokens debugging simple errors. Enable "diffs" feature to save tokens.

***

### Lovable (formerly GPT Engineer)
Lovable positions itself as the world's first "AI Fullstack Engineer," building complete web applications from plain English prompts up to 20 times faster than traditional development. The platform generates modern tech stacks using React and Tailwind CSS on the frontend with Supabase backend integration, targeting non-technical founders, product managers, and developers who want to skip boilerplate setup. Lovable emphasizes rapid MVP development with GitHub sync, custom domains, and deployment capabilities built in.

**Pricing:**
- Free: $0 (5 credits/day, up to 30/month max, public projects only)
- Pro: $25/month (100 monthly credits + 5 daily credits, up to 150 total/month, credit rollovers)
- Business: $50/month (100 monthly credits + 5 daily, SSO, data opt-out, design templates)
- Enterprise: Custom pricing (dedicated support, custom connections, group access control)
- Student discount: Up to 50% off with university email

**Key features:**
- Natural language to full-stack web app generation
- React + Tailwind CSS + Supabase tech stack
- GitHub synchronization for version control
- Custom domain support and instant deployment
- Real-time collaboration for Pro and Teams plans
- Unlimited public projects on free tier

**Status:** Active and rapidly evolving (formerly GPT Engineer open-source project with 50k+ GitHub stars). Major limitation is credit-based pricing where every interaction including debugging consumes message credits, making costs unpredictable for complex projects. Users report "wasting credits" on debugging loops. Free plan very restrictive (exhausted without building production project). Best for quick prototypes and MVPs rather than production development.

***

### Replit
Replit is an AI-powered cloud development platform offering full-stack app generation through Replit Agent, real-time multiplayer collaboration, and integrated cloud services. The platform supports 50+ programming languages with both browser-based and desktop IDE options, targeting developers from hobbyists to enterprise teams.

**Pricing:**
- Starter: Free (10 development apps, public access only, 2 GiB storage, limited AI access)
- Core: $20/month annually ($25 monthly) - includes $25 monthly credits, unlimited AI access, Claude Sonnet 4 & GPT-4o, 4 vCPUs, 8 GiB memory, 50 GiB storage
- Teams: $35/user/month - includes $40 credits/user, 50 viewer seats, 8 vCPUs, 16 GiB memory, 250 GiB storage, private deployments
- Enterprise: Custom pricing (SSO, SCIM, single tenant, custom viewer seats, dedicated support)
- Optional add-ons: Always-on apps, Boosts, Deployments (autoscale starting at $1/month, reserved VM at $20/month)

**Key features:**
- AI-powered full-stack app generation (Replit Agent)
- Real-time multiplayer collaboration
- Integrated cloud services (databases, storage, authentication)
- Mobile app for iOS and Android development
- Browser-based and desktop IDE options
- Support for 50+ programming languages

**Status:** Active with recent pricing model change (July 2025 to dynamic usage-based pricing). Major user complaints about new pricing—reports of $350 charges in single day due to agent performing excessive loops. Standard agent now starts executing immediately even for simple questions, with poor instruction following. Opus 4 model available performs better but significantly more expensive. Users advised to monitor costs carefully per checkpoint. Best for learning and experimentation on Free/Core; use Teams/Enterprise cautiously with cost monitoring.

***

### Replit
Replit is an AI-powered cloud development platform offering full-stack app generation through Replit Agent, real-time multiplayer collaboration, and integrated cloud services. The platform supports 50+ programming languages with both browser-based and desktop IDE options.

**Pricing:**
- Starter: Free (10 development apps, public access, 2 GiB storage/app, 10 GiB data transfer, limited AI)
- Core: $20/month annually - Full Replit Agent access, $25 flexible credits/month, Claude Sonnet 4 & GPT-4o, unlimited AI access, 4 vCPUs, 8 GiB memory, 50 GiB storage, 100 GiB data transfer, priority support
- Teams: $35/user/month - Everything in Core + $40 credits/user/month, 50 viewer seats, 8 vCPUs, 16 GiB memory, 250 GiB storage, 1,000 GiB data transfer, private deployments, role-based access
- Enterprise: Custom pricing - SSO/SAML, SCIM, single tenant GCP, custom connections, dedicated support

**Key features:**
- AI-powered full-stack app generation (Replit Agent)
- Real-time multiplayer collaboration
- Integrated cloud services (databases, storage, authentication)
- Mobile app for iOS and Android development
- Browser-based and desktop IDE options
- Support for 50+ programming languages

**Status:** Active with significant pricing controversy following July 2025 switch to dynamic usage-based model. Users report extreme costs ($350/day) due to agent performing excessive loops even for simple queries. Agent struggles with instruction following, requiring extremely precise prompts. Opus 4 model available is better but much more expensive. Critical issue: switching from high-power to standard agent may continue charging at high-power rate (possible bug). Best for hobbyists on Free plan; Core plan viable for controlled usage; Teams/Enterprise require careful cost monitoring.

---

## Application: Frontend/Component Development (Native AI Features)

### V0 by Vercel
V0 by Vercel is an AI-powered frontend component generator created by the makers of Vercel and Next.js, designed to accelerate UI development through text prompts. Unlike full-stack builders, V0 focuses specifically on generating React and Tailwind CSS components that integrate seamlessly with Next.js applications. The platform emphasizes design system consistency and production-ready code quality, with generated components that can be copied directly into projects or deployed via Vercel's hosting infrastructure.

**Pricing:**
- Free: $5 monthly credits for limited deployments
- Premium: $20/month with $20 monthly credits and 5x larger file support
- Team: $30/user/month with shared credits and collaboration tools
- Enterprise: Custom pricing with SSO and priority support

**Key features:**
- AI generation of React + Tailwind components from prompts
- Visual design mode for editing without code
- Figma import capabilities for design-to-code workflows
- GitHub sync for version control integration
- API access for automation and workflow integration
- Next.js optimization and Vercel hosting integration

**Status:** Active with ongoing development. Best for developers working within the Next.js/Vercel ecosystem who need quick UI component scaffolding. Free tier is generous for experimentation and learning. V0 is more focused than full-stack AI builders, making it ideal for frontend-heavy projects rather than complete application development.

***

### Builder.io
Builder.io is a visual development platform combining AI-powered design-to-code conversion with a headless CMS and visual editor for creating web experiences. The platform offers two complementary products: "Fusion" for AI-assisted development and code generation, and "Publish" for visual content management and page building. Builder.io targets enterprise teams needing both rapid development capabilities and content management with version control, A/B testing, and personalization features.

**Pricing:** Contact for pricing (enterprise-focused)

**Key features:**
- AI design-to-code conversion from Figma/Sketch
- Visual drag-and-drop editor for components
- Headless CMS with scheduling and workflows
- A/B testing and personalization engine
- Version control and collaboration tools
- Integration with React, Next.js, and major frameworks
- API access for custom implementations
- Multi-language and localization support

**Status:** Active and enterprise-focused. Best for larger teams needing both visual development capabilities and robust content management with A/B testing. Positioned as a hybrid between visual builder and headless CMS.

---

## Application: Website Building (Native/Hybrid AI Features)

### Webflow
Webflow is a mature visual web design platform that has integrated AI capabilities into its established no-code website builder. While not purely AI-driven like newer tools, Webflow combines powerful visual design tools with recent AI features for site generation, content creation, and optimization. The platform serves designers, agencies, and marketing teams who need pixel-perfect custom websites without writing code, offering CMS functionality, e-commerce capabilities, and hosting in a single platform.

**Pricing:**
**Site Plans:**
- Free/Starter: $0 (2 pages, 50 CMS items, .webflow.io domain)
- Basic: $14/month annually ($18 monthly) - custom domain, no CMS
- CMS: $23/month annually ($29 monthly) - 20 collections, 2,000 items, 50GB bandwidth
- Business: $39/month annually ($49 monthly) - 40 collections, 10,000 items, 100GB bandwidth
- Enterprise: Custom pricing

**E-commerce Plans:**
- Standard: $29/month annually ($42 monthly) - 500 items, 2% transaction fee
- Plus: $74/month annually ($84 monthly) - 5,000 items, 0% transaction fee
- Advanced: $212/month annually ($235 monthly) - 15,000 items, 0% transaction fee

**Workspace Plans:**
- Freelancer: $16/month annually
- Agency: $35/month annually (phased out, replaced by Growth at $49/month)
- Full Seat: $39/month per user
- Limited Seat: $15/month per user (content editors)

**Add-ons:**
- Localization: From $9/month per locale
- Analyze: From $9/month
- Optimize: From $299/month (A/B testing)

**Key features:**
- Visual design editor with pixel-perfect control
- AI Site Builder (Beta) for rapid site generation from prompts
- Built-in CMS with up to 40 collections and 10,000 items
- Native e-commerce with up to 15,000 products
- Designer-friendly interactions and animations
- SEO tools and performance optimization
- Team collaboration and client billing features

**Status:** Active and mature platform with ongoing AI feature development. Pricing increased moderately for 2025. Legacy Editor Seats being phased out by end of 2025, requiring transition to new Workspace system. Annual billing saves ~20-25%. Best for designers, agencies, and marketing teams needing professional websites with visual control. Most popular plan is CMS ($23/month annually) for content-driven sites.

***

### Framer
Framer is an AI-powered web design and prototyping tool that combines a visual builder with natural language site generation, targeting designers and non-technical founders who want production-ready websites quickly. Framer's AI can generate complete responsive websites from text prompts, which designers can then refine using the visual editor with drag-and-drop components. The platform emphasizes speed and design quality, offering built-in CMS, hosting, and interactions without requiring separate plugins or add-ons.

**Pricing:**
**Site Plans (per site):**
- Free: $0 (Framer subdomain, "Made in Framer" banner, 10 CMS collections with 1,000 items)
- Mini: $5/month annually ($10 monthly) - 1 page + 404, 1 GB bandwidth, 1,000 visitors, custom domain
- Basic: $15/month annually ($20 monthly) - 150 pages, 10 GB bandwidth, 10,000 visitors, 1 CMS collection with 1,000 items
- Pro: $30/month annually ($40 monthly) - 300 pages, 100 GB bandwidth, 200,000 visitors, 10 CMS collections with 10,000 items, global CDN
- Scale: $100/month annually - Higher limits, A/B testing capability, reverse proxy support
- Enterprise: Custom pricing

**Editor Seats (Workspace):**
- Personal plans: $20/month per additional editor ($15 annually)
- Business plans: $40/month per editor (up to 10 editors)

**Add-ons:**
- Localization: $20/month per locale for Personal ($15 annually); $40/month for Business (includes AI translations)
- Advanced Analytics: Available on Business plans
- Custom Proxy: Available on Scale plan

**Key features:**
- AI text-to-website generation with instant results
- Visual editor for refining AI-generated designs
- Built-in CMS with up to 10,000 items
- Native animations and interactions
- Responsive design automation
- Custom code support for advanced functionality
- Team collaboration and version control
- SEO optimization and fast hosting included

**Status:** Active with generous student program. Site plans separate from Editor seats (site owner always free). Most popular: Basic ($15/month annually) for personal/small business sites, Pro ($30/month annually) for high-traffic/complex sites. Recent pricing changes controversial (localization now $40/month per language for Business plans). Best for designers wanting AI-assisted site generation with visual refinement capabilities.

***

### Dora
Dora is a native AI 3D website builder that generates 3D animated websites from text prompts. The platform features a no-code editor for complex 3D interactions, generative 3D interaction capabilities, and advanced AI animation features with drag-and-connect constraint layout system.

**Pricing:** Contact for pricing (free plan available for testing)

**Key features:**
- AI generation of 3D animated websites from text prompts
- No-code editor for complex 3D interactions
- Generative 3D interaction capabilities
- Advanced AI animation features
- Drag-and-connect constraint layout system
- Instant publishing with SSL included
- Figma plugin available

**Status:** Active and innovative. Positioned as unique in offering 3D website generation. Best for designers and creative professionals wanting to create visually striking 3D websites without coding. Relatively new to market compared to Webflow/Framer.

***

### WeWeb
WeWeb is an AI-driven no-code platform for building web apps with a visual drag-and-drop interface using Vue.js components. The platform offers full code export capabilities (PWA) and extensive backend integration options.

**Pricing:** Contact for pricing (multiple tiers available)

**Key features:**
- AI-driven no-code platform for web apps
- Visual drag-and-drop interface with Vue.js components
- Full code export capabilities (PWA)
- Backend integration with REST, GraphQL, Supabase, Xano, and more
- Advanced business logic and access control
- Custom JavaScript, HTML, and CSS support

**Status:** Active with recent 3.0 update (February 2025) focused on future of no-code. User reports indicate platform is "no-code but not low-effort" - requires technical understanding despite visual interface. Best for teams comfortable with technical concepts who want rapid web app development with backend flexibility.

---

## Application: Web & Mobile App Development (No-Code Platforms)

### Bubble
Bubble is a full-featured no-code platform for building web applications with complex database structures, workflows, and user authentication without writing code. Unlike simpler site builders, Bubble targets serious application development with visual programming for business logic, API integrations, and scalable backend infrastructure. The platform uses a usage-based pricing model centered on "Workload Units" (WUs) that measure server capacity, making it essential to understand resource consumption to control costs.

**Pricing:**
- Free: $0 (50,000 WU/month, Bubble branding, Bubble domain, SOC Type 2 compliance)
- Starter: $29/month annually ($32 monthly) - 175,000 WU/month, custom domain, remove branding
- Growth: $119/month annually ($134 monthly) - 250,000 WU/month, additional features
- Team: $269/month annually ($329 monthly) - 350,000 WU/month, collaboration features
- Enterprise: Custom pricing
- Overage rate: $0.30 per 1,000 WU (lower rates available with workload tier subscriptions)
- Mobile app builder: Separate pricing as of October 2025

**Key features:**
- Visual programming for complex application logic
- Database design with relational data structures
- User authentication and role-based permissions
- API connector for third-party integrations
- Responsive design editor
- Custom workflows and conditional logic
- Version control and deployment management
- Native mobile app builder (separate pricing)

**Status:** Active with controversial shift to Workload Units pricing model (October 2024). This consumption-based model charges for actual backend resource usage, similar to traditional coding environments. Users must carefully optimize workflows to control costs - inefficient database queries or frequent API calls can rapidly consume WU. Each action (data fetch, workflow, file upload, page load, API call) consumes WU based on complexity. Best for serious app development where technical understanding of optimization is available. Free plan good for learning; paid plans require ongoing cost monitoring.

***

### FlutterFlow
FlutterFlow is a visual development platform for building native mobile applications using Google's Flutter framework without writing code. The platform enables rapid iOS and Android app development through drag-and-drop interfaces, pre-built templates, and extensive integration options with backends like Firebase and Supabase. FlutterFlow underwent significant pricing restructuring in August 2025, transitioning to team-based plans that align with how developers actually scale applications.

**Pricing:** (Restructured August 2025 to team-based plans)
- Free: Basic features with limitations
- Standard: Contact for current pricing post-August 2025 restructure
- Pro: Contact for current pricing
- Teams: Contact for current pricing (emphasizes team collaboration)
- Enterprise: Custom pricing

**Key features:**
- Visual Flutter app builder with 1,000+ templates
- Native iOS and Android app generation
- Firebase, Supabase, and custom API integration
- Source code export and GitHub integration
- AI-assisted development and code generation
- Custom code support for advanced functionality
- App store deployment tools (APK/IPA generation)
- Branching for version control and collaboration

**Status:** Active with major pricing restructure in August 2025 moving to team-based plans. Change reflects shift toward how developers scale apps collaboratively. Best for mobile app development teams wanting native iOS/Android apps with Flutter framework. Source code export allows moving off platform if needed.

***

### Glide
Glide is a no-code platform for building mobile-friendly Progressive Web Apps (PWAs) from spreadsheets, with integration to Google Sheets, Airtable, BigQuery, and SQL databases.

**Pricing:** Contact for current pricing (multiple tiers available)

**Key features:**
- Progressive Web Apps (PWAs) from spreadsheets
- Integration with Google Sheets, Airtable, BigQuery, SQL
- Glide Big Tables for high-scale data
- Native AI actions for text, audio, and image processing
- Install as PWA on devices
- Real-time data synchronization

**Status:** Active with some user concerns about platform lock-in and scalability. Reddit users report "might've been a mistake" building on Glide for production apps. Best for quick internal tools and data-driven apps from spreadsheets rather than complex production applications. PWA approach means no native app store distribution.

***

### Adalo
Adalo is a no-code mobile and web app builder featuring drag-and-drop interface for creating iOS, Android, and web applications with direct publishing to App Store and Google Play.

**Pricing:** Multiple tiers available (contact for current pricing)

**Key features:**
- Drag-and-drop builder for iOS, Android, and web apps
- Internal database or integration with Xano, Airtable
- Direct publishing to App Store and Google Play
- Responsive design across all devices
- Component marketplace with integrations
- Zapier integration for workflow automation

**Status:** Active with positive user reviews for ease of use. Considered one of the easiest no-code platforms for beginners. Best for entrepreneurs and non-technical founders wanting to build mobile apps quickly. Native app publishing capability distinguishes from PWA-only platforms.

***

### Softr
Softr is a no-code web app builder for creating web and progressive web apps (PWAs) with extensive database integrations (14+ sources including Airtable, PostgreSQL, Supabase).

**Pricing:** Multiple tiers available (contact for current pricing)

**Key features:**
- Web and progressive web app (PWA) creation
- Database integrations (14+ sources including Airtable, PostgreSQL, Supabase)
- AI tools for app generation, text, and images
- SEO functionalities with automatic indexing
- Permission system for user groups
- Block-building system with specialized templates

**Status:** Active with generally positive user experiences. February 2025 announcement expanded beyond Airtable databases to broader integration options. Best for creating client portals, internal tools, and membership sites with existing database backends.

***

### Retool
Retool is a low-code platform specifically designed for building internal tools and data-driven applications for business operations. Unlike consumer-facing app builders, Retool focuses on admin panels, dashboards, CRUD applications, and workflow tools that connect to databases, APIs, and business systems. The platform provides pre-built UI components optimized for internal use cases, extensive data connectors, and the ability to write custom JavaScript/SQL when needed.

**Pricing:** Contact for pricing (multiple tiers including on-premise)

**Key features:**
- 100+ pre-built UI components for internal tools
- Native connections to databases (PostgreSQL, MySQL, MongoDB, etc.)
- REST API and GraphQL integration
- Custom JavaScript and SQL support
- Mobile app builder for iOS/Android internal apps
- Role-based access control and permissions
- Version control and deployment workflows
- On-premise deployment option

**Status:** Active and enterprise-focused. Best for companies needing to build custom internal tools quickly. Strong database connectivity and ability to write custom code when visual builder insufficient. Positioned for technical teams building admin panels and operational tools rather than customer-facing apps.

---

## Application: Design Prototyping (Native No-Code Features)

### Uizard
Uizard is an AI-powered design tool that transforms text prompts, hand-drawn sketches, and wireframes into high-fidelity UI mockups and prototypes for web and mobile applications. The platform emphasizes accessibility for non-designers through its "Autodesigner" feature that generates complete app designs from text descriptions, along with screenshot-to-mockup conversion and wireframe enhancement capabilities. Uizard targets product managers, founders, and designers who need rapid iteration without deep design expertise.

**Pricing:** Contact for current pricing (multiple tiers available)

**Key features:**
- AI generation from text prompts, sketches, or screenshots
- Autodesigner for complete app/website mockup creation
- Drag-and-drop editor with 100+ templates
- Heatmap prediction for UX optimization
- Real-time collaboration and commenting
- Design handoff with specs for developers
- AI-powered theming and component suggestions
- Export to PNG, PDF, and code (HTML/CSS)

**Status:** Active and focused on democratizing design for non-designers. Best for product managers and founders who need to visualize ideas quickly before engaging designers. AI capabilities make it accessible to non-technical users. Code export allows handoff to developers.

---

## Application: Code Editing (AI-First IDEs)

### Cursor
Cursor is an AI-first code editor built as a fork of VS Code, designed to accelerate professional development through integrated AI assistance. Unlike no-code builders, Cursor targets experienced developers who want AI to enhance their existing coding workflow rather than replace it. The platform provides multi-file context awareness, AI-powered debugging, and codebase-wide understanding through integration with GPT-4, Claude, and other models.

**Pricing:** Contact for current pricing

**Key features:**
- AI-first IDE with VS Code compatibility
- Multi-file context and codebase understanding
- AI-assisted debugging and refactoring
- Natural language to code generation
- Inline code suggestions and completions
- Terminal command generation
- Git integration and version control
- Support for all major programming languages

**Status:** Active and popular among professional developers. Positioned as enhancing coding workflow rather than replacing it. Best for developers comfortable with code who want AI assistance for faster development, debugging, and refactoring.

***

### Windsurf
Windsurf is an AI-native IDE with Cascade agent, featuring Supercomplete for intent prediction, in-editor live previews, and one-click deploys from the IDE.

**Pricing:** Contact for current pricing

**Key features:**
- AI-native IDE with Cascade agent
- Supercomplete for intent prediction
- In-editor live previews of frontend
- One-click deploys from IDE
- Model Context Protocol (MCP) support
- Natural language commands in code and terminal
- Available on Mac, Windows, and Linux

**Status:** Active and positioned as next-generation AI IDE. Competes with Cursor in AI-first code editor space. Features like Cascade agent and Supercomplete aim to predict developer intent before full prompting. Best for developers wanting highly integrated AI assistance throughout coding workflow.

***

### GitHub Copilot
GitHub Copilot is a non-native AI code assistant plugin that integrates with major IDEs including VS Code, JetBrains, and Neovim.

**Pricing:** 
- Individual: ~$10/month
- Business: ~$19/user/month
- Enterprise: ~$39/user/month

**Key features:**
- AI code completion plugin for major IDEs
- Context-aware code suggestions
- Support for dozens of programming languages
- Integration with VS Code, JetBrains, Neovim
- GitHub integration for repository context

**Status:** Active and widely adopted. First major AI coding assistant to reach mainstream adoption. Best for developers using existing IDEs who want AI code suggestions without switching editors. Integration with GitHub provides strong repository context awareness.

---

## Application: Backend Development (Backend-as-a-Service Platforms)

### Supabase
Supabase is an open-source Backend-as-a-Service platform built on PostgreSQL, offering managed database, authentication, real-time subscriptions, file storage, and edge functions.

**Pricing:**
- Free: $0 (500MB database, 1GB file storage, 50MB file uploads, 2GB bandwidth)
- Pro: ~$25/month (8GB database, 100GB file storage, 5GB file uploads, 250GB bandwidth)
- Team: Contact for pricing
- Enterprise: Custom pricing (SLA, support, on-premise options)

**Key features:**
- Managed PostgreSQL database
- Built-in authentication (email/password, OAuth, magic links)
- Real-time database updates via WebSockets
- File storage for images, videos, documents
- Edge Functions for serverless logic
- Auto-generated APIs for database queries
- Open-source with self-hosting option

**Status:** Active and rapidly growing as Firebase alternative. Open-source nature allows self-hosting and prevents vendor lock-in. Best for developers wanting PostgreSQL-based backend with real-time capabilities. Strong developer community. Positioned as "open source Firebase alternative."

***

### Xano
Xano is a no-code backend platform featuring visual API builder, PostgreSQL database management, and auto-generated RESTful APIs.

**Pricing:** 
- Launch: Contact for pricing
- Scale: Contact for pricing
- Enterprise: Custom pricing

**Key features:**
- Visual no-code API builder
- Scalable PostgreSQL database management
- Auto-generated RESTful APIs
- Business logic with no-code function stack
- User authentication and role-based access control
- Serverless hosting with auto-scaling
- Third-party API integrations and webhooks

**Status:** Active with some user concerns. Reddit users advise "read this before picking Xano" due to limitations and scaling challenges. Best for no-code app builders (FlutterFlow, Adalo, etc.) needing custom backend logic beyond what Firebase/Supabase provide visually. Requires understanding of API concepts despite no-code approach.

***

### Firebase (Google)
Firebase is Google's Backend-as-a-Service platform offering real-time NoSQL database (Firestore), authentication, cloud storage, and serverless functions.

**Pricing:**
- Spark (Free): Generous free tier
- Blaze (Pay as you go): Usage-based pricing
- Enterprise: Custom pricing

**Key features:**
- Real-time NoSQL database (Firestore)
- Authentication with multiple providers
- Cloud storage for files
- Cloud Functions for serverless backend logic
- Analytics and crash reporting
- Push notifications
- Machine learning integration

**Status:** Active and mature platform from Google. Most established BaaS option with extensive documentation and community. NoSQL (Firestore) vs. SQL (Supabase) is key differentiator. Best for mobile apps and real-time applications needing Google ecosystem integration. Free tier very generous for learning and small projects.

---

## Application: Workflow Automation (Native Automation Platforms)

### Make (formerly Integromat)
Make is a native automation platform featuring visual workflow builder with integration to 1,500+ apps.

**Pricing:** Multiple tiers from free to enterprise

**Key features:**
- Visual workflow builder
- Integration with 1,500+ apps
- Advanced logic and conditional routing
- Error handling and debugging
- API integration capabilities

**Status:** Active and popular among technical users. More complex but more powerful than Zapier. Best for users needing advanced automation logic and complex workflows.

***

### Zapier
Zapier is a native automation platform connecting 5,000+ apps without code, featuring multi-step workflow automation (Zaps).

**Pricing:** 
- Free: Limited Zaps and tasks
- Starter: ~$20/month
- Professional: ~$50/month
- Team: ~$300/month
- Company: ~$600/month

**Key features:**
- Connect 5,000+ apps without code
- Multi-step workflow automation (Zaps)
- Conditional logic and filters
- Built-in formatter and utilities
- Version control for workflows

**Status:** Active and market leader in no-code automation. Easiest to use but most expensive for high-volume automation. Best for business users needing simple app integrations without technical knowledge.

***

### n8n
n8n is a native open-source automation platform offering self-hosted workflow automation with 350+ integrations.

**Pricing:**
- Self-hosted: Free (open-source)
- Cloud: Starting ~$20/month

**Key features:**
- Self-hosted workflow automation
- Visual workflow builder
- 350+ integrations
- Custom code nodes (JavaScript)
- Git-based version control for workflows
- Open-source with cloud hosting option

**Status:** Active open-source project. Best for technical teams wanting self-hosted automation with code customization. Lower integration count than Zapier/Make but open-source and self-hostable. Appeals to privacy-conscious organizations and those wanting workflow version control in Git.

---

## Application: Visual Development (Native Visual Builders)

### Plasmic
Plasmic is a visual builder that allows designing in Plasmic and rendering in any codebase, featuring Figma import capabilities.

**Pricing:** Free tier available with paid plans for teams

**Key features:**
- Design in Plasmic, render in any codebase
- Figma import capabilities
- CMS integration
- Component library system
- Git-based version control for designs
- Export to React, Vue, or other frameworks

**Status:** Active with focus on design-to-code workflow. Unique positioning allowing visual design that integrates into developer codebases. Best for teams wanting visual design capabilities while maintaining code-based projects.

---

## Application: Mobile App Development (Native No-Code Platforms)

### Appgyver (SAP Build Apps)
Appgyver, now SAP Build Apps, is a native no-code platform for building cross-platform mobile and web apps.

**Pricing:** Contact SAP for enterprise pricing

**Key features:**
- Cross-platform mobile and web apps
- Visual development environment
- Integration with SAP systems
- Data binding and logic builders
- Native device capabilities access

**Status:** Active as part of SAP ecosystem. Best for enterprises already using SAP systems needing custom mobile apps integrated with SAP data and workflows.

---

## Application: Database/Backend (Native Hybrid Platforms)

### Airtable
Airtable is a native spreadsheet-database hybrid offering spreadsheet interface with database capabilities and API generation.

**Pricing:**
- Free: Limited records and features
- Plus: ~$10/user/month
- Pro: ~$20/user/month
- Enterprise: Custom pricing

**Key features:**
- Spreadsheet interface with database capabilities
- API generation from tables
- Automation workflows
- Integration with 1,000+ apps via Zapier
- Collaboration and permissions
- Version history and snapshots

**Status:** Active and popular for non-technical database needs. Often paired with no-code front-ends (Softr, Glide, etc.) as backend. Best for teams wanting database flexibility with spreadsheet familiarity. Limitations with high-volume data compared to traditional databases.
