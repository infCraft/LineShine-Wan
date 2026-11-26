#!/usr/bin/env python3
"""Build the SME Linear bridge in a compute node's local /tmp directory."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    if not build.is_relative_to(Path("/tmp")):
        parser.error("--build-dir must be on the compute node under /tmp")
    if "0xd22" not in Path("/proc/cpuinfo").read_text():
        parser.error("build on a 72F5 compute node")
    from torch.utils.cpp_extension import load
    build.mkdir(parents=True, exist_ok=True)
    load(name="xhpops_biasfix", sources=[str(Path(__file__).with_name("xh_linear_op.cpp"))],
         extra_cflags=["-O3", "-march=armv9-a+sme+bf16+sve"],
         extra_ldflags=["-ldl"], with_cuda=False, verbose=True,
         is_python_module=False, build_directory=str(build))
    print("XH_LINEAR_BUILD_OK", build / "xhpops_biasfix.so", flush=True)


if __name__ == "__main__":
    main()
