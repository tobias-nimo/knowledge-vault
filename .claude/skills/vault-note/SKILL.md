---
name: vault-note
description: Use when the user pastes source material (a book, paper, or course excerpt) to be turned into a note in this Obsidian knowledge vault, or asks to create/add/expand a vault note. Renders the note in the user's established writing style and places it in the correct folder with proper figures, tags, and wiki-links.
---

# Vault Note

Turn raw source material (book/paper/course excerpts) into a note that matches this vault's established style, and place it correctly. The goal is a **paraphrased, distilled** note — never a verbatim copy of the source.

## Workflow

1. **Identify placement.** Notes live at `source/Topic/Subtopic/Note Title.md`, where `source` is `books/`, `papers/`, or `courses/`. Match the existing folder for the topic; create a new subfolder only when the topic clearly warrants one. Confirm the target path with the user if it's ambiguous.
   - **Cross-cutting concept notes** (not tied to one chapter) go in a per-source `notes/` folder, grouped by category: `notes/concepts/` (e.g. `Sampling Bias.md`, `No Free Lunch Theorem.md`), `notes/sklearn/` (e.g. `MNIST.md`, `Imputers.md`), `notes/miscellaneous/`. These are the usual targets of `[[wiki-links]]` from chapter notes.
2. **Read a sibling note first.** Before writing, read 1–2 existing notes in the same folder to match their depth, section granularity, and conventions. Styles are consistent across the vault but density varies by topic.
3. **Distill, don't transcribe.** Paraphrase the source in the user's voice. Drop page numbers, figure-caption boilerplate, and footnote markers. Keep the technical substance, code, and key numbers.
4. **Write in the style below.**
5. **Wire it up.** Add figures, `#tags`, and `[[wiki-links]]` per the conventions below.
6. **Don't touch `INDEX.md` / `README.md`** unless the user asks, or you're adding an entirely new top-level source.

## Writing style

- **Title.** First line is `# Title` (H1), matching the filename. No frontmatter in vault notes.
- **Opening.** Start with a one-line definition of the concept, **bolding the key term or the core idea**. Example: `#overfitting means that the model **performs well on the training data, but it does not generalize well**.`
- **Bold liberally** on key terms and important phrases — this is the primary emphasis device. Use *italics* sparingly for secondary terms or names being introduced.
- **Voice.** Second person and explanatory ("you can…", "let's…"). Concise and direct; explain the *why*, not just the *what*.
- **Sections.** Use `##` for sections and `###` for subsections. Short notes may have no sections at all (a single flowing explanation is fine — see `No Free Lunch Theorem.md`).
- **Lists.** Use bullet lists for enumerations and solution sets; nest when there's a natural hierarchy.

### Code

- Fenced ```python blocks. Show expected output as an inline `#` comment, e.g. `cross_val_score(...) # array([0.95035, 0.96035, 0.9604])`.
- Keep imports that aid clarity; trim plotting boilerplate to the essential lines and replace the rest with `plt.show()`.
- Use inline `#` comments to annotate non-obvious lines. Avoid markdown backticks inside code comments (they don't render).
- For REPL-style output that's a multi-line array, a plain (non-python) fenced block is acceptable — match `Performance Measures.md`.
- **Don't re-dump shared dataset setup.** When a sibling note already defines the dataset/variables, import them with a pointer comment instead of repeating the `fetch_openml`/split boilerplate, e.g. `from my_datasets import X_train, y_train, some_digit # defined in ./performance-measures`.

### Math

- LaTeX with `$$…$$` for display and `$…$` inline. Wrap words/labels in `\text{}` (e.g. `\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}`).
- Define each symbol in a short bullet list right after the formula.

### Callouts and asides

- `>` blockquote for a short remark, rule of thumb, or "remember" aside.
- Obsidian callouts for emphasis: `>[!tip]`, `>[!warning]`, `>[!note]`. Use `[!warning]` for pitfalls/caveats, `[!tip]` for guidance and rules of thumb, and `[!note]` for neutral asides or behavior remarks (e.g. tie-breaking rules). A callout can have a title on the same line as the type.

## Figures

- Embed with `![[name.png]]` (Obsidian embed syntax), on its own line where the source places the figure.
- Figures live in a **per-book/source `figures/` folder**, named by the source's own numbering (e.g. `3-4.png`, `1-23.png`).
- Captions/explanations of a figure go in a `>` blockquote **immediately below** the embed, not in the embed itself.
- If a referenced figure file doesn't exist yet, still add the `![[…]]` embed and **tell the user it's missing** so they can drop the image in.

## Tags and wiki-links

- **Tags:** add an inline `#kebab-case` tag on the **first mention** of a key concept the note is about (e.g. `#sklearn`, `#overfitting`, `#feature-engineering`, `#precision-recall-trade-off`). Don't tag every term — only the note's salient concepts. A tag can sit mid-sentence as the term itself (`#recall is the ratio…`).
- **Wiki-links:** link to sibling notes with `[[Note Title]]` when referencing a concept that has (or should have) its own note (e.g. `[[Sampling Bias]]`, `[[MNIST]]`). Linking to a note that doesn't exist yet is fine — it marks it for later. Match the exact title of the target note.
- External sources get standard markdown links: `[text](url)`.

## Quick checklist before finishing

- [ ] H1 title matches filename; no verbatim copying from the source.
- [ ] Key term bolded in the opening line.
- [ ] Placed in the correct `source/Topic/Subtopic/` folder.
- [ ] Figures embedded with `![[…]]`; missing ones flagged.
- [ ] `#tags` on first mention of salient concepts; `[[links]]` to related notes.
- [ ] Code outputs shown as inline comments; math symbols defined.
