# GGUF Tools

Random tools and experiments for manipulating or viewing the GGUF file format. See:

1. https://github.com/ggerganov/llama.cpp
2. https://github.com/ggerganov/ggml

## Scripts

All scripts support using `--help` for information on the commandline options.

### `gguf-checksum`

Allows calculating a model's SHA256 without being affected by the exact order of the fields in the file. It's
also possible to get checksums of the individual tensors or metadata fields.

### `gguf-frankenstein`

You supply an input metadata GGUF file and optionally an input tensor data GGUF file and this utility
will stitch the two together into a new GGUF file. When the tensor data file isn't specified, you
end up with a vocab-only model that just has the metadata. This could be used for future Frankenstein-ing
or training a model with that vocab/metadata as the base.

## Disclaimer

These scripts are experimental and likely not very well tested. They may or may not work. Use at your own risk.
