# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-04

### Added

- Initial release of omnivoice-server
- OpenAI-compatible TTS API (`/v1/audio/speech`)
- Three voice modes:
  - Auto: Model selects voice automatically
  - Design: Specify voice attributes (gender, age, accent, etc.)
  - Clone: Voice cloning from reference audio
- Voice profile management API (`/v1/voices/profiles`)
  - Create, read, update, delete voice cloning profiles
  - Persistent storage for reusable voice profiles
- One-shot voice cloning endpoint (`/v1/audio/speech/clone`)
- Streaming synthesis support (sentence-level chunking)
- Model listing endpoint (`/v1/models`)
- Health check endpoint (`/health`)
- Metrics endpoint (`/metrics`)
- CLI interface with `omnivoice-server` command
- Configuration via environment variables or CLI flags
- Optional Bearer token authentication
- Concurrent request handling with configurable limits
- Request timeout protection
- Audio format support: WAV and raw PCM
- Speed control (0.25x - 4.0x)
- Configurable inference steps (1-64)
- Python client examples
- cURL examples
- Streaming audio player example
- Comprehensive documentation
- CI/CD workflow with GitHub Actions

### Technical Details

- Built on FastAPI and Uvicorn
- Uses OmniVoice model from k2-fsa
- Supports CUDA, MPS, and CPU inference
- Thread pool executor for concurrent synthesis
- Pydantic-based configuration and validation
- Type hints throughout codebase
- Async/await for I/O operations

[unreleased]: https://github.com/maemreyo/omnivoice-server/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/maemreyo/omnivoice-server/releases/tag/v0.1.0
