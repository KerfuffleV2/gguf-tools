#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from gguf.constants import GGUFValueType, Keys
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter


def go(args: argparse.Namespace) -> None:
    print(f"* Loading metadata source: {args.metadata}")
    md_reader = GGUFReader(args.metadata, "c")
    if args.tensors is not None:
        print(f"* Loading tensor data source: {args.metadata}")
        td_reader: None | GGUFReader = GGUFReader(args.tensors, "c")
    else:
        td_reader = None
        print("* No tensor data source specified, output will be a vocab-only model.")
    arch = str(
        md_reader.fields[Keys.General.ARCHITECTURE].parts[-1].tobytes(),
        encoding="utf-8",
    )
    gw = GGUFWriter(args.output, arch, use_temp_file=False)
    print("* Adding metadata: ", end="")
    for kv in md_reader.fields.values():
        if (
            kv.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count")
            or not kv.types
        ):
            # Not real metadata fields.
            continue
        if kv.name == Keys.General.ARCHITECTURE:
            # Gets added automatically.
            continue
        print(f"{kv.name!r}, ", end="")
        sys.stdout.flush()
        if len(kv.types) > 2 or (
            len(kv.types) == 2 and kv.types[1] == GGUFValueType.ARRAY
        ):
            msg = f"Cannot handle types in field {kv.name!r}: {kv.types}"
            raise ValueError(msg)
        if kv.types[0] == GGUFValueType.ARRAY:
            itype = kv.types[1]
            gw.add_key(kv.name)
            chunk = BytesIO()
            chunk.write(gw._pack("I", GGUFValueType.ARRAY))
            chunk.write(gw._pack("I", itype))
            chunk.write(gw._pack("Q", len(kv.data)))
            if itype == GGUFValueType.STRING:
                for di in kv.data:
                    sval = kv.parts[di].tobytes()
                    chunk.write(gw._pack("Q", len(sval)))
                    chunk.write(sval)
            else:
                for di in kv.data:
                    chunk.write(kv.parts[di].tobytes())
            gw.kv_data += chunk.getvalue()
            del chunk
            gw.kv_data_count += 1
        elif kv.types[0] == GGUFValueType.STRING:
            gw.add_string(
                kv.name,
                str(
                    kv.parts[kv.data[0]].tobytes(),
                    encoding="utf-8",
                ),
            )
        else:
            gw.add_key(kv.name)
            gw.add_val(kv.parts[kv.data[0]][0], kv.types[0])
    print()
    if td_reader is not None:
        print("* Adding tensor metadata: ", end="")
        for tensor in td_reader.tensors:
            print(f"{tensor.name!r}, ", end="")
            gw.add_tensor_info(
                tensor.name,
                [int(val) for val in reversed(tensor.shape)],
                np.dtype(np.float32),
                int(tensor.n_bytes),
                raw_dtype=tensor.tensor_type,
            )
        print()
    print("* Writing metadata")
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    if td_reader is not None:
        gw.write_ti_data_to_file()
        print("* Writing tensor data")
        n_tensors = len(td_reader.tensors)
        for idx, tensor in enumerate(td_reader.tensors, 1):
            prettysize = f"{tensor.n_bytes / (1024 * 1024):.2f}"
            print(
                f"  {idx:4}/{n_tensors:4} | type:{tensor.tensor_type.name:6} | size:{prettysize:>8} MiB | name:{tensor.name}",
            )
            gw.write_tensor_data(tensor.data)
    gw.close()
    print("* Successful completion.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model combining utility for GGUF files",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=str,
        help="GGUF format file to use for metadata",
    )
    parser.add_argument(
        "--tensors",
        type=str,
        help="GGUF format file to use for tensor data, if not specified you'll end up with a vocab-only result",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="GGUF format output file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting the output file if it already exists",
    )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    if args.output in (args.metadata, args.tensors):
        print("! Cannot overwrite input files with --out", file=sys.stderr)
        sys.exit(1)
    if not args.force and Path(args.output).exists():
        print("! Output file already exists", file=sys.stderr)
        sys.exit(1)
    go(args)


if __name__ == "__main__":
    main()
