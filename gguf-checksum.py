#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from hashlib import sha256
from typing import Any

from gguf.gguf_reader import GGUFReader, ReaderField, ReaderTensor


def go(args: argparse.Namespace) -> None:
    reader = GGUFReader(args.model, "r")
    metadata_hash = sha256()
    kv_hash = sha256()
    tensor_info_hash = sha256()
    tensor_data_hash = sha256()
    overall_hash = sha256()
    result: dict[str, Any] = {}
    if args.hash_individual_kvs:
        result["kvs"] = {}
    if args.hash_individual_tensors:
        result["tensors"] = {}
    if not args.skip_kvs:

        def kv_sortfun(field: ReaderField) -> str:
            return field.name

        kvs = sorted(reader.fields.values(), key=kv_sortfun)
        for field in kvs:
            if field.name.startswith("__SPARE"):
                continue
            curr_kv_hash = sha256()
            for part in field.parts:
                pb = part.tobytes()
                metadata_hash.update(pb)
                kv_hash.update(pb)
                if args.hash_individual_kvs:
                    curr_kv_hash.update(pb)
                overall_hash.update(pb)
            if args.hash_individual_kvs:
                if args.json:
                    result["kvs"][field.name] = curr_kv_hash.hexdigest()
                else:
                    print(f"HASH {'KV':15} {curr_kv_hash.hexdigest()} {field.name!r}")
    if not args.skip_tensors:

        def tensor_sortfun(tensor: ReaderTensor) -> str:
            return tensor.name

        tis = sorted(reader.tensors, key=tensor_sortfun)
        for tensor in tis:
            # Skip the offset
            for part in tensor.field.parts[:-1]:
                pb = part.tobytes()
                metadata_hash.update(pb)
                tensor_info_hash.update(pb)
                overall_hash.update(pb)
        if not args.skip_tensor_data:
            for tensor in tis:
                tb = tensor.data.tobytes()
                tensor_data_hash.update(tb)
                overall_hash.update(tb)
                if args.hash_individual_tensors:
                    tensor_hash = sha256()
                    tensor_hash.update(tb)
                    if args.json:
                        result["tensors"][tensor.name] = tensor_hash.hexdigest()
                    else:
                        print(
                            f"HASH {'TENSOR':15} {tensor_hash.hexdigest()} {tensor.name!r}",
                        )
                elif not args.json:
                    print(
                        f"HASH {'RUNNING':15} {overall_hash.hexdigest()} {tensor.name!r}",
                    )
    if not args.skip_kvs:
        if args.json:
            result["kv_metadata"] = kv_hash.hexdigest()
        else:
            print(f"HASH {'KV_METADATA':15} {kv_hash.hexdigest()}")
    if not args.skip_tensors:
        if args.json:
            result["tensor_metadata"] = tensor_info_hash.hexdigest()
            if not args.skip_tensor_data:
                result["tensor_data"] = tensor_data_hash.hexdigest()
        else:
            print(f"HASH {'TENSOR_METADATA':15} {tensor_info_hash.hexdigest()}")
    if not (args.skip_kvs and args.skip_tensors):
        if args.json:
            result["metadata"] = metadata_hash.hexdigest()
        else:
            print(f"HASH {'METADATA':15} {metadata_hash.hexdigest()}")
    if not args.skip_tensor_data:
        if args.json:
            result["tensor_data"] = tensor_data_hash.hexdigest()
        else:
            print(f"HASH {'TENSOR_DATA':15} {tensor_data_hash.hexdigest()}")
    if args.json:
        import json

        result["overall"] = overall_hash.hexdigest()
        json.dump(result, sys.stdout)
    else:
        print(f"HASH {'OVERALL':15} {overall_hash.hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checksum utility for GGUF files")
    parser.add_argument("model", type=str, help="GGUF format model filename")
    parser.add_argument(
        "--skip-kvs",
        action="store_true",
        help="Skip KV metadata in result",
    )
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="Skip tensors in result, implies --skip-tensor-data as well",
    )
    parser.add_argument(
        "--skip-tensor-data",
        action="store_true",
        help="Skip tensor data in result",
    )
    parser.add_argument(
        "--hash-individual-kvs",
        action="store_true",
        help="Include individual KV hashes in result, no effect when --skip-kvs specified",
    )
    parser.add_argument(
        "--hash-individual-tensors",
        action="store_true",
        help="Include individual tensor data hashes in result, no effect when --skip-tensors or --skip-tensor-data specified",
    )
    parser.add_argument("--json", action="store_true", help="Produce JSON output")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    go(args)


if __name__ == "__main__":
    main()
