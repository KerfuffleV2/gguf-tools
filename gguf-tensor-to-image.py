#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
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


def make_image(tensor: ReaderTensor) -> Image.Image:
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
    td = (255 * ((td - np.min(td)) / np.ptp(td))).astype(np.uint8)
    mode = "L"
    return Image.fromarray(td, mode)


def go(args: argparse.Namespace) -> None:
    reader = GGUFReader(args.model, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    names = tensors.keys() if args.tensor == ["*"] else args.tensor
    for tk in names:
        tensor = tensors[tk]
        if "/" in tensor.name:
            raise ValueError("Bad tensor name")
        if len(names) == 1:
            output = args.output
        else:
            filepath = args.output.parent
            filename = args.output.name
            output = filepath / f"{tensor.name}.{filename}"
        print(
            f"* Saving tensor {tensor.name!r} ({tensor.tensor_type.name}) to: {output}",
        )
        img = make_image(tensor)
        img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checksum utility for GGUF files")
    parser.add_argument(
        "model",
        type=str,
        help="GGUF format model filename",
    )
    parser.add_argument(
        "tensor",
        nargs="+",
        type=str,
        help="Tensor name. You may use * to match all tensors, but do not specify multiple tensor arguments in that case",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file, will be prefixed with the tensor name if multiple tensor names are specified",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwriting the output file if it already exists",
    )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    go(args)


if __name__ == "__main__":
    main()
