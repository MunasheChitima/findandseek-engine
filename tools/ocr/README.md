# `ocrtool` — the macOS OCR helper

A ~180-line Swift program that wraps Apple's Vision framework. It is the
production OCR path on macOS: on-device, Neural-Engine accelerated, zero-egress,
and roughly 100× faster per scanned page than running a vision model. On Linux
this is unused — OCR goes through tesseract instead.

`ocrtool.swift` is the whole source. `ocrtool` is a compiled `arm64` binary,
committed so that `uv sync` gives you working OCR without needing Xcode.

## If you'd rather not run a binary you didn't build

That's a reasonable position, and you don't have to. Build it yourself and point
the engine at your copy:

```bash
swiftc -O tools/ocr/ocrtool.swift -o /tmp/ocrtool
export FINDANDSEEK_OCR_BIN=/tmp/ocrtool
```

`FINDANDSEEK_OCR_BIN` is read at import time by
[`find_and_seek/ingest/vision_macos.py`](../../find_and_seek/ingest/vision_macos.py),
so nothing else needs changing. Requires the Xcode command-line tools
(`xcode-select --install`); the frameworks it links — Foundation, Vision,
AppKit — all ship with macOS.

To skip the helper entirely, set `FINDANDSEEK_ENABLE_OCR=0`. Images and scanned
PDFs then index on filename and metadata alone, with no text extraction.

## Why there is no checksum here

A published hash would only tell you that the file you cloned is the file that
was committed. It could not tell you that the binary corresponds to the source
next to it — Swift builds are not byte-reproducible, so compiling `ocrtool.swift`
yourself will *not* produce a matching hash even if the source is identical and
untampered. Publishing a checksum would imply a guarantee it can't provide.

The honest options are the real ones: read the 180 lines of Swift and build it
yourself, or run the committed binary. Both are supported; neither is pretended
to be the other.

For reference, the committed binary is:

```
sha256  726eeac6cb61914acc8e9bad94126b899490bf9f5d6a054820ed567426ff5183
        arm64 Mach-O executable
```

Useful for confirming a download wasn't corrupted, or that the file hasn't
changed between two checkouts. Not evidence about its contents.

## What it does

```
ocrtool <img1> [img2 ...]   batch mode — one delimited block per image
ocrtool --server            stream mode — one image path per stdin line
```

The engine uses `--server`: the helper is launched once and kept alive, so each
image costs a write and a read rather than a process spawn. Two Vision requests
run per image — `VNRecognizeTextRequest` for text, and `VNClassifyImageRequest`
for scene tags, the latter only on low-text images so that text pages don't
collect noisy labels.

A per-image timeout (`FINDANDSEEK_OCR_TIMEOUT`, default 20s) bounds the call; if
the helper doesn't return its end sentinel in time it is killed and ingest moves
on, so a single undecodable image can't wedge the worker.
