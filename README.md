# grounded-semantic-pipeline
Grounded Semantic Failures in SLM-Based Cross-Lingual Dialogue Summarization

This repository supports a diagnostic study of hallucination and factuality
failures in English-to-Chinese dialogue summarization with small language
models (SLMs).

The main goal is to determine whether summary hallucinations arise from
source-side grounded semantic understanding failures before target-language
generation begins. The project compares an mBART baseline and direct SLM
summarization, diagnoses summary-level hallucination, probes source-side
understanding through structured semantic extraction, and links representation
failures to downstream summary errors.

## Project Stages

1. Baseline comparison: expand the test set, run mBART and direct SLM pipelines,
   and compare ROUGE, BERTScore, OmniScore, and output length.
2. Summary-level diagnosis: analyze hallucination and faithfulness errors in
   direct Chinese summaries.
3. Representation-level diagnosis: run grounded semantic extraction on the same
   English source dialogues.
4. Failure categorization: categorize failures in events, roles, evidence
   grounding, and final outcome prediction.
5. Propagation analysis: map semantic extraction failures to downstream
   hallucination and faithfulness errors.

See `docs/project_structure.md` for the proposal-aligned folder layout.

