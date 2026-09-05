# TinyStories selected dataset — data notice

Original data providers: Ronen Eldan and Yuanzhi Li, TinyStories.
Source: https://huggingface.co/datasets/roneneldan/TinyStories
Revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
Paper: https://arxiv.org/abs/2305.07759

This dataset is provided under the unmodified **Community Data License Agreement — Sharing, Version 1.0 (CDLA-Sharing-1.0)**:
https://cdla.dev/sharing-1-0/

The upstream dataset card at the pinned revision identifies this data license. The MIT license of M³'s application code does not replace the dataset's license.

M³ changes: expansion to the first 200,000 unique training stories, excluding the parent held-out story hashes; whitespace-normalized exact deduplication across selected splits; byte-for-byte preservation of the parent 500 validation and 500 test stories; JSONL serialization; reuse of the parent 2,048-token byte-level BPE tokenizer fitted only on the original 20k training stories; conversion into little-endian uint16 token streams with EOS separators. See manifest.json for source URLs, counts, selection rules and file hashes. Story text is preserved except for stripping outer whitespace and normalizing file serialization. The archive's per-file .NOTICE sidecars identify these transformations. No claim of authorship of the original stories is made.
