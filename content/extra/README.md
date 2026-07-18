# Add your own documents here

Drop `.md` files in this folder, then rebuild the index:

```bash
./index.sh
```

## Chunking

Each file is split into chunks by `##` (level-2) headings:

- One chunk per `##` section (plus the intro before the first heading).
- Files with no `##` headings stay a single chunk.
- Very short sections are merged into the previous one.

So write long articles with clear `##` sections, or keep short notes as one file.

Optional frontmatter at the top improves source citations:

```markdown
---
title: My Article Title
source_url: https://original-source.com/article
topic: career change
---

# My Article Title
...
```

Example topics: career advice, portfolio tips, interview prep, CV guidance.
