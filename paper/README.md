# Paper source

LaTeX source for *Relational Grounding Failures in Zero-Shot Cross-Lingual Dialogue Summarization with Small Language Models* (EMNLP workshop submission).

## Build

```bash
cd paper/latex
pdflatex acl_latex.tex
bibtex acl_latex
pdflatex acl_latex.tex
pdflatex acl_latex.tex
```

Main entry point: `acl_latex.tex`. Figures are in `figures/`.

After deanonymization, switch `\usepackage[review]{acl}` to `\usepackage[final]{acl}` and update author metadata.
