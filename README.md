# grounded-semantic-pipeline
Grounding Failures in SLM-Based Cross-Lingual Dialogue Summarization

This repository supports a diagnostic study of hallucination and factuality
failures in English-to-Chinese dialogue summarization with Small Language
Models (SLMs). As SLMs become increasingly important for efficient, accessible,
and locally deployable NLP systems, it is crucial to understand not only whether
they can generate fluent cross-lingual summaries, but also whether their outputs
are faithfully grounded in the source dialogue.

We evaluate SLMs in a zero-shot direct setting, where each model is prompted to
generate a Chinese summary directly from an English dialogue. These outputs are
compared against an mBART baseline, but the goal is diagnostic rather than
performance-oriented. Instead of training or fine-tuning models, we analyze how
existing SLMs fail as grounded cross-lingual dialogue summarizers.

The study begins with automatic evaluation using ROUGE, BERTScore, and
OmniScore, then focuses on fine-grained hallucination and grounding errors that
are not fully captured by aggregate metrics. To support this analysis, the
project introduces a grounding-oriented error taxonomy for cross-lingual
dialogue summarization. The taxonomy distinguishes hallucination errors
involving entities or participants, events and actions, speaker roles,
circumstantial details, negation and modality, and discourse-level outcomes. It
also separately tracks supported over-generation, salient omissions, and
language or format issues, since these reflect related but distinct
output-quality failures.

The central hypothesis is that hallucinations in zero-shot cross-lingual
dialogue summarization are not random surface errors, but systematic failures in
dialogue grounding. In particular, SLMs may misrepresent who participated in the
dialogue, what events occurred, who did what to whom, whether an event was
certain or negated, and what final outcome the conversation reached. By
analyzing these patterns across models, this project aims to show where SLMs
remain fragile as grounded cross-lingual summarizers and why fluent Chinese
summaries can still be unfaithful to the English source dialogue.

## Project Stages

1. Baseline comparison: run the mBART baseline and zero-shot direct SLM
   pipelines on the same English-to-Chinese dialogue summarization test set.
2. Automatic evaluation: compare ROUGE, BERTScore, OmniScore, and output length
   across models.
3. Summary-level diagnosis: analyze hallucination and faithfulness errors in the
   generated Chinese summaries.
4. Grounding-oriented categorization: label failures involving entities,
   events, speaker roles, circumstantial details, negation/modality, and
   discourse-level outcomes.
5. Related quality analysis: separately track supported over-generation, salient
   omissions, and language or format issues.
6. Cross-model pattern analysis: identify systematic grounding vulnerabilities
   across SLMs and compare them with the mBART baseline.

See `docs/project_structure.md` for the proposal-aligned folder layout.
