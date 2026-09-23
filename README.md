# Omni-Analyst: Multi-Agent AI Verification System

[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)
[![dual-license](https://img.shields.io/badge/dual--license-AGPL--3.0--only%20or%20commercial-blueviolet)](LICENSING.md)

## License

This project is dual-licensed under **AGPL-3.0-only** OR a commercial license.

- [LICENSE](LICENSE) — GNU AGPL-3.0-only (the free track)
- [LICENSING.md](LICENSING.md) — how the two tracks work
- [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) — the commercial agreement
- [NOTICE](NOTICE) — copyright, SPDX identifier, and provenance


A 9-phase protocol for AI research with built-in cross-verification to 
eliminate hallucination.

## The Problem
Current AI systems hallucinate facts with no verification layer. This 
causes legal liability, reputational damage, and unreliable research.

## The Solution
Multiple specialized AI agents collaborate to verify information before 
output:
- Search agents find sources
- Analysis agents extract claims
- Verification agents cross-check claims against sources
- Synthesis agents produce reports from verified data only

## Proof of Concept
The `proof-of-concept/` directory contains a working PyTorch simulation 
demonstrating the core verification logic using a Dissonance Resolution 
Architecture.

## Architecture
See `architecture/` for complete technical specifications.

## Status
Architecture complete. Seeking technical co-founder for implementation.

## Contact
[samgrim97@gmail.com]
