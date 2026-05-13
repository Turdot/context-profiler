# Security Policy

## Reporting Vulnerabilities

Please report security issues privately by opening a GitHub security advisory or contacting the maintainer.

Do not include private traces, credentials, API keys, customer data, or production prompts in public issues.

## Trace Data Safety

Agent traces often contain sensitive data:

- API keys and bearer tokens
- customer messages
- source code
- internal file paths
- prompts and tool schemas
- stack traces and environment variables

Before sharing examples publicly, redact or synthesize trace content. Prefer minimal synthetic fixtures that reproduce the behavior being discussed.

## Project Boundary

`context-profiler` is a local analysis tool. It does not upload traces or call remote APIs by default.

Future integrations should preserve this boundary unless explicitly documented and opt-in.
