# Viewable data versions

Each directory is an immutable dataset version. Its `manifest.json` identifies the upstream revision, selected story counts, split rules and SHA-256 hashes of all data files. `tokenizer.json` is the exact fitted tokenizer; `version.json` links the version to its local/mounted artifacts. Git commits show additions and changes to these small records. The dashboard links each version to its GitHub files and upstream source and displays paginated story content from the mounted artifact directory.

The raw JSONL stories and token arrays remain under ignored `data/`. They are not uploaded by a Git push. Back up that directory separately and mount it on the Linux training host. The manifest lets any restored/downloaded copy be checked. A source link is not an independent backup: until object storage is configured, the full selected dataset is available locally and through its upstream source only.

Register a prepared version with `python tools/register_dataset.py --data data/VERSION --name VERSION`. Registration refuses to replace a prior version. Dataset edits must create a new version, be reviewed in Git, and be tied to a new experiment. Tokenizer or held-out split changes must be explicit; do not compare their perplexity scores as if the protocols were identical.
