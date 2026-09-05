# Local dashboard and later Linux deployment

Start from the repository root:

```sh
.venv/bin/python -m dashboard.server --device cpu --port 8765
```

Open http://127.0.0.1:8765. CPU inference is the default so a running MPS training process can use the GPU; select `--device mps` or `--device cuda` explicitly when appropriate. The dashboard reads `runs/` and registered `datasets/` and loads checkpoints directly. It does not call an external model API or launch training jobs.

Experiments show training/validation curves, exact configuration, source commit, dataset hash, runtime/cost records and all saved fixed-prompt samples. Active status is inferred from recent metrics; stale/unknown means the process cannot be confirmed, not that it completed. The dashboard is read-only except for inference requests; it cannot approve, launch, stop or delete runs.

Playground supports story completion and a chat-style story history. Both use the chosen base checkpoint. The model is not instruction-tuned; the interface does not claim general conversational ability. A response uses at most 256 new tokens and the checkpoint's context window; older history is trimmed and the UI reports it. One inference request runs at a time; the model is cached until a newer checkpoint is saved. Prompts/history are kept in the browser's page memory and are not persisted by the server.

Data versions show committed manifests/tokenizers, original source links and a paginated story browser. The selected raw file is checked against its registered hash before display. The test split is sealed in the browser during model selection; a downloaded full archive still includes it for reproducibility. Archive checksums are checked before downloading, and the restore tool verifies all files after unpacking. Dataset archives are local until `published_url` records a successful remote upload.

## Linux/cloud layout

The same Python application runs on CPU, native Mac MPS or Linux CUDA. Both Docker definitions include the dashboard and small registered metadata. Mount `data/` and `runs/` separately; do not bake large datasets or checkpoints into images. Use persistent storage for those mounts. Export the selected dataset archive, verify its committed checksum, and restore the same relative layout on Linux.

For initial remote use, keep the server bound to loopback and use SSH port forwarding:

```sh
# On the Linux host, from the checked-out repository:
python -m dashboard.server --device cuda --host 127.0.0.1 --port 8765

# On your Mac (replace the host):
ssh -L 8765:127.0.0.1:8765 user@your-gpu-host
```

The local URL then reaches the GPU host without publishing an internet endpoint. Inside a container, use host-network mode on Linux to retain this loopback-only arrangement, or bind `0.0.0.0` inside the container with token protection and publish only `127.0.0.1:8765:8765` on the host.

Non-loopback binding requires `MCUBED_DASHBOARD_TOKEN` of at least 24 characters. API requests must send it as a Bearer token; the UI asks for it and stores it in sessionStorage for that tab. `--allowed-host` adds an explicitly permitted hostname. Cross-origin generation requests are rejected. For public deployment, add TLS and a production reverse proxy/authentication layer; this small standard-library HTTP server is intended for one-user local or tunneled use. Never bake tokens into the image or commit them to Git.

The future cloud dashboard can use the same API contract and artifacts. GPU scheduling, durable chat history, training approvals and cloud billing enforcement remain separate milestones.
