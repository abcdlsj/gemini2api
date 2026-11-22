# Standalone binary

Kip looks for one release asset per platform plus a manifest. The contract is small on
purpose: the gemini2api plugin in Kip downloads the artifact for its platform, verifies the
sha256 from `manifest.json`, and only then runs it.

```sh
uv sync
uv pip install pyinstaller
uv run --no-sync pyinstaller --noconfirm --clean packaging/gemini2api.spec
./dist/gemini2api --help      # or: ./dist/gemini2api cookies --write
```

Artifacts are named `gemini2api-<os>-<arch>` (`.exe` on Windows), where `<os>` is
`macos`, `win`, or `linux`, and `<arch>` is `arm64` or `x64`. The release also carries a
`manifest.json`:

```json
{
  "version": "0.1.0",
  "artifacts": {
    "macos-arm64": {
      "file": "gemini2api-macos-arm64",
      "url": "https://github.com/abcdlsj/gemini2api/releases/download/v0.1.0/gemini2api-macos-arm64",
      "sha256": "…",
      "size": 20148736
    }
  }
}
```

Kip downloads the artifact for its platform, verifies `sha256`, and only then runs it. The
binary keeps the normal CLI: with no arguments it serves; `cookies` and `doctor` are
subcommands so a frozen build needs no `python -m`.
