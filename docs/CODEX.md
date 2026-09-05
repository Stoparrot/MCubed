# Codex context for M³

Open this folder as the project and select **GPT-6 Astra** in the task's model picker. Use high reasoning for experiment design and code review; lower effort is sufficient for routine command execution. The runtime does not call an OpenAI API and needs no OpenAI API key. Codex plan usage remains separate from training costs.

`AGENTS.md` contains the stable project instructions. Keep it small. The requirements define the purpose; the experiment file defines the scientific contract; `STATUS.md` carries current facts between tasks. Do not paste entire datasets or log files into the prompt. Ask Codex to read the relevant artifacts instead.

Suggested next task prompt:

> Read AGENTS.md and docs/STATUS.md. Act as Engram-μ and Model creator for experiment 001. Inspect the pilot artifacts, explain what they establish, and run the baseline in configs/baseline.json on MPS within its existing local time budget. Keep the tokenizer, data split and architecture fixed. Use validation for decisions; leave test data untouched until the final baseline is selected. Record results and remaining limitations. Continue ordinary implementation and validation work without repeated confirmation.

For later concept work, replace the objective with one observable behavior and specify its controls, metric, threshold and budget. Let the experiment designer own parameter choices; separate the hypothesis from implementation edits.

Use Git branches and review commits before formal runs. Future dashboard launch approvals must identify an exact commit and artifacts. A chat approval of a hypothesis is not an unlimited cloud-spend authorization.

On this Mac, `/usr/bin/git` initially encountered an unaccepted full-Xcode license. The installed Command Line Tools work with:

```sh
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
```

This is a shell-local toolchain selection. No license was accepted and no global Xcode setting was changed. Native MPS is available, but Codex's sandbox may hide GPU access; the GPU doctor and training commands may need the app's execution approval. Do not silently record a CPU run as a GPU result.

References: [project instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [GPT-6 Astra guidance](https://developers.openai.com/api/docs/guides/latest-model). The setup deliberately uses project context rather than global model or permission changes.
