#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

try:
    from PIL import Image
except ImportError:
    print("This script requires Pillow installed. Example: pip install pillow")
    sys.exit(1)

from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader, ReaderTensor

# Clip values to at max 7 standard deviations from the mean.
CFG_SD_CLIP_THRESHOLD = 7

# Number of standard deviations above the mean to be positive scaled.
CFG_SDP_THRESHOLD = 1.25

# Number of standard deviations below the mean to be negative scaled.
CFG_SDN_THRESHOLD = 1.25

# RGB scaling for pixels that meet the negative threshold.
CFG_NEG_SCALE = (1.0, 0.6, 0.7)

# RGB scaling for pixels that meet the positive threshold.
CFG_POS_SCALE = (0.6, 1.0, 0.7)

# RGB scaling for pixels between those ranges.
CFG_MID_SCALE = (0.5, 0.5, 0.8)


class Quantized:
    dtype: np.dtype[Any]
    block_size: int

    def quantize(self, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        raise NotImplementedError("Ohno")

    def dequantize(self, arr: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        raise NotImplementedError("Ohno")


class Quantized_Q8_0(Quantized):  # noqa: N801
    block_size = 32
    dtype = np.dtype([("d", "f2"), ("qs", "i1", (block_size,))])

    # Mini Q8_0 quantization in Python!
    @classmethod
    def quantize(cls, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        n_blocks = arr.size // cls.block_size
        blocks = arr.reshape((n_blocks, cls.block_size))

        # Much faster implementation of block quantization contributed by @Cebtenzzre
        def quantize_blocks(blocks: npt.NDArray[Any]) -> Iterable[tuple[Any, Any]]:
            d = abs(blocks).max(axis=1) / np.float32(127)
            with np.errstate(divide="ignore"):
                qs = (blocks / d[:, None]).round()
            qs[d == 0] = 0
            yield from zip(d, qs)

        return np.fromiter(
            quantize_blocks(blocks),
            count=n_blocks,
            dtype=cls.dtype,
        )

    @classmethod
    def dequantize(
        cls,
        arr: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.float32]:
        blocks = arr.view(dtype=cls.dtype)
        return (blocks["d"][:, None] * np.float32(blocks["qs"])).flatten()


def make_image(args: argparse.Namespace, tensor: ReaderTensor) -> Image.Image:
    if len(tensor.shape) > 2:
        raise ValueError("Can only handle 1d and 2d tensors")
    td: np.ndarray[Any, Any]
    if tensor.tensor_type == GGMLQuantizationType.F16:
        td = tensor.data.view(dtype=np.float32)
    elif tensor.tensor_type == GGMLQuantizationType.F32:
        td = tensor.data
    elif tensor.tensor_type == GGMLQuantizationType.Q8_0:
        td = Quantized_Q8_0.dequantize(tensor.data).reshape(tensor.shape)
    else:
        raise ValueError("Cannot handle tensor type")
    if len(tensor.shape) == 1:
        if args.adjust_1d_rows is not None:
            td = td.reshape((args.adjust_1d_rows, td.shape[0] // args.adjust_1d_rows))
        else:
            td = td[None, :]
    mode = args.mode
    if mode == "devs-overall":
        sd = np.std(td)
        mean = np.mean(td)
    elif mode == "devs-rows":
        sd = np.std(td, axis=1)[:, None]
        mean = np.mean(td, axis=1)[:, None]
    elif mode == "devs-cols":
        sd = np.std(td, axis=0)
        mean = np.mean(td, axis=0)
    else:
        raise ValueError("Unknown mode")
    sdp_max = mean + CFG_SD_CLIP_THRESHOLD * sd
    sdp_thresh = mean + CFG_SDP_THRESHOLD * sd
    sdn_thresh = mean - CFG_SDN_THRESHOLD * sd
    tda = np.minimum(np.abs(td), sdp_max).repeat(3, axis=-1).reshape((*td.shape, 3))
    tda = 255 * ((tda - np.min(tda)) / np.ptp(tda))
    tda[td <= sdn_thresh, ...] *= CFG_NEG_SCALE
    tda[td >= sdp_thresh, ...] *= CFG_POS_SCALE
    tda[np.logical_and(td > sdn_thresh, td < sdp_thresh), ...] *= CFG_MID_SCALE
    return Image.fromarray(tda.astype(np.uint8), "RGB")


def go(args: argparse.Namespace) -> None:
    print(f"* Loading: {args.model}")
    reader = GGUFReader(args.model, "r")
    tensors: OrderedDict[str, ReaderTensor] = OrderedDict(
        (tensor.name, tensor) for tensor in reader.tensors
    )
    if args.match_glob:
        names = [
            name
            for name in tensors
            if any(fnmatch.fnmatchcase(name, pat) for pat in args.tensor)
        ]
    elif args.match_regex:
        res = [re.compile(r) for r in args.tensor]
        names = [name for name in tensors if any(r.search(name) for r in res)]
    else:
        names = [tensors[name].name for name in args.tensor]
    print(f"* Matching tensors: {', '.join(repr(n) for n in names)}")
    for tk in names:
        tensor = tensors[tk]
        output: None | Path = None
        if "/" in tensor.name:
            raise ValueError("Bad tensor name")
        if args.output is not None:
            if len(names) == 1:
                output = args.output
            else:
                filepath = args.output.parent
                filename = args.output.name
                output = filepath / f"{tensor.name}.{filename}"
        print(
            f"* Processing tensor {tensor.name!r} (type:{tensor.tensor_type.name}, shape:{tensor.shape})",
        )
        img = make_image(args, tensor)
        if args.scale != 1.0:
            img = img.resize(
                (int(img.width * args.scale), int(img.height * args.scale)),
                resample=Image.Resampling.LANCZOS,
            )
        if output is not None:
            print(f"-  Saving to: {output}")
            img.save(output)
        if args.show_with:
            print("-  Displaying to screen")
            if output is not None:
                subprocess.call((args.show_with, output))  # noqa: S603
            else:
                with tempfile.NamedTemporaryFile(suffix=".png") as fp:
                    img.save(fp, format="png")
                    fp.flush()
                    subprocess.call((args.show_with, fp.name))  # noqa: S603


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tensor to image converter for GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Information on output modes:
              devs-*:
                overall: Calculates the standard deviation deviation from the mean.
                         By default, values below the mean will be red and values above it will be green.
                rows   : Same as above, except the calculation is based on rows.
                cols:  : Same as above, except the calculation is based on columns.
        """,
        ),
    )
    parser.add_argument(
        "model",
        type=str,
        help="GGUF format model filename",
    )
    parser.add_argument(
        "tensor",
        nargs="+",
        type=str,
        help="Tensor name, may be specified multiple times UNLESS --match-glob or --match-regex is used",
    )

    output_group = parser.add_argument_group(
        "output",
        "At least one of the following must be specified:",
    )
    output_group.add_argument(
        "--output",
        type=Path,
        help="Output file, will be prefixed with the tensor name if multiple tensor names are specified",
    )
    output_group.add_argument(
        "--show-with",
        help="""
            Show the result with the specified application.
            WARNING: If processing multiple tensors and your image application
            does not block then you will end up with a bunch of huge images displayed at the same time""",
    )

    wildcard_group = parser.add_mutually_exclusive_group()
    wildcard_group.add_argument(
        "--match-glob",
        action="store_true",
        help="Interpret tensor name as a glob, so wildcards like blk.0.* will work",
    )
    wildcard_group.add_argument(
        "--match-regex",
        action="store_true",
        help="Interpret tensor name as a regex, so regular expressions like ^blk\\.[012]\\.attn will work",
    )

    parser.add_argument(
        "--match-1d",
        action="store_true",
        help="When using a wildcard, also match 1 dimensional tensors",
    )
    parser.add_argument(
        "--adjust-1d-rows",
        type=int,
        help="""
        Instead of rendering 1D tensors as a wide image with one row, rearrange into multiple rows.
        For example, if we have a 1D tensor 3,200 elements and specify "--adjust-1d-rows 32",
        the output image will have dimensions 100x32. Note: The tensor size must be divisible by
        the specified value.
        """,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale the image. Default: 1.0 (no scaling)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting the output file if it already exists",
    )
    parser.add_argument(
        "--mode",
        choices=["devs-overall", "devs-rows", "devs-cols"],
        default="devs-overall",
        help="Output modes (see below). Default: devs-overall",
    )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    if not (args.show_with or args.output):
        print("! At least one of --show or --output must be specified", file=sys.stderr)
        sys.exit(1)
    if (args.match_regex or args.match_glob) and len(args.tensor) != 1:
        print(
            "! Can only specify one tensor name (pattern) when using --match-glob or --match-regex",
            file=sys.stderr,
        )
    go(args)
    print("\n* Done.")


if __name__ == "__main__":
    main()
