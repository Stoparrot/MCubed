# Viewable data versions

Each directory is an immutable dataset version. Its `manifest.json` identifies the upstream revision, selected story counts, split rules and SHA-256 hashes of all data files. `tokenizer.json` is the exact fitted tokenizer; `version.json` links the version to its local/mounted artifacts. Git commits show additions and changes to these small records. The dashboard links each version to its GitHub files and upstream source and displays paginated story content from the mounted artifact directory.

The raw JSONL stories and token arrays remain under ignored `data/`. They are not uploaded by a Git push. Back up that directory separately and mount it on the Linux training host. The manifest lets any restored/downloaded copy be checked. A source link is not an independent backup: until object storage is configured, the full selected dataset is available locally and through its upstream source only.

The exact local dataset can also be packaged as a deterministic compressed archive under `data/archives/`. Its checksum and location are committed in `version.json`; the archive itself is not an ordinary Git blob. `tools/dataset_archive.py` packs it and restores only known regular files after checking the archive checksum, then verifies each restored file. The dashboard offers the local archive for download. No remote archive upload is implied by registration or packaging; `published_url` is null until a release/storage upload actually succeeds.

TinyStories data retains **CDLA-Sharing-1.0** and its original attribution. Read the version's `DATA_NOTICE.md`; M³ application code is MIT.

Register a prepared version with `python tools/register_dataset.py --data data/VERSION --name VERSION`. Registration refuses to replace a prior version. Dataset edits must create a new version, be reviewed in Git, and be tied to a new experiment. Tokenizer or held-out split changes must be explicit; do not compare their perplexity scores as if the protocols were identical.

Restore a transferred archive from the repository root (choose a new destination):

```sh
.venv/bin/python tools/dataset_archive.py restore --name tinystories-20k --archive data/archives/tinystories-20k-ba66e1b75b1b.tar.gz --out data/restored-20k
```

The archive is kept local at the user's request; no GitHub Release is created.
