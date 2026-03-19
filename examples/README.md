# Examples

End-to-end demo: convert external data → profiler input → analysis report.

## Supported Input Formats

context-profiler natively accepts these formats (auto-detected):

| Format | Key Signature | Mode |
|--------|--------------|------|
| **OpenAI** | `{messages: [{role, content}], tools: [{type: "function", function: {...}}]}` | snapshot |
| **Anthropic** | `{messages: [{role, content: [{type: "text"/"tool_use"/"tool_result", ...}]}], tools: [{name, input_schema}]}` | snapshot |
| **Langfuse trace** | `{id, observations: [{type: "GENERATION", input: {messages, tools}}]}` | session |
| **JSONL** (`.jsonl`) | One OpenAI/Anthropic request per line | session |
| **Directory** | Folder of `.json` files, each one request | session |

## Demo: Toolathlon Trajectory → Report

[Toolathlon](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories) is a public dataset with 5000+ agent execution trajectories from 17 LLMs. Its format is **not** directly compatible — this demo shows how to convert and analyze it.

### Raw data

`toolathlon_raw.json` — a GPT-5 agent executing a train ticket planning task:
- 22 messages, 40 tool definitions, 11 LLM calls
- ~80K total tokens across all calls

### Step 1: Convert to snapshot (single request)

```bash
cd examples/

# Convert to a single OpenAI-format request (final context window)
python convert_toolathlon.py toolathlon_raw.json --mode snapshot -o snapshot.json

# Analyze
context-profiler analyze snapshot.json
```

### Step 2: Convert to session (per-call snapshots)

```bash
# Reconstruct each LLM call as a separate snapshot → JSONL
python convert_toolathlon.py toolathlon_raw.json --mode session -o session.jsonl

# Analyze with session mode (shows context growth timeline)
context-profiler analyze session.jsonl --html report.html
```

### What you get

**Snapshot mode** — token distribution of the final (most bloated) API call:
- How much of the context is tool definitions vs actual conversation
- Which tools consume the most tokens

**Session mode** — context growth timeline across all 11 LLM calls:
- How the message token count grows from call to call
- Tool definitions stay constant while messages accumulate

## Adapting Other Formats

The same pattern works for any external format — write a small script that reshapes the data into OpenAI `{messages, tools}` format:

| Source Format | Key Difference | Conversion |
|--------------|----------------|------------|
| **ShareGPT** | `conversations`/`from`/`value` | Rename to `messages`/`role`/`content`, map `human`→`user`, `gpt`→`assistant` |
| **LMSYS / WildChat** | `conversation` instead of `messages` | Rename field |
| **Toolathlon** | Flat message array + stringified fields | Deserialize strings, split on assistant boundaries for session |
| **LangSmith** | Hierarchical runs with `inputs`/`outputs` | Extract `run_type=llm` runs, use `inputs.messages` |
