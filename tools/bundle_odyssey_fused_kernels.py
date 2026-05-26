"""Bundle pre-built dropout_layer_norm + fused_dense_lib .so files into a single
``odyssey_fused_kernels`` wheel.

Both subdirectories of flash-attention build their own .so as standalone wheels.
Odyssey wants one URL pin in its pyproject, so we extract the .so files from
each subdir wheel and repackage them into a single wheel whose top-level layout
is just the two .so files plus dist-info metadata. Python's ``import`` does not
care what wheel a .so came from, so ``import dropout_layer_norm`` and
``import fused_dense_lib`` continue to work unchanged for callers.

Filename convention mirrors flash-attn's existing builds:

    odyssey_fused_kernels-<version>+<build-tag>-cp<py>-cp<py>-linux_<arch>.whl

where ``build-tag`` is e.g. ``cu128torch2.9``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import platform
import re
import sys
import sysconfig
import zipfile
from pathlib import Path


PACKAGE_NAME = "odyssey_fused_kernels"
DIST_INFO_DESCRIPTION = (
    "Bundled, pre-built flash-attention subdir kernels (dropout_layer_norm + "
    "fused_dense_lib) for Odyssey lab/post. Top-level .so layout — no Python "
    "wrapper module. Built by .github/workflows/build-odyssey-fused-wheels.yml "
    "on the odysseyml/flash-attention fork."
)


def _record_entry(member_name: str, data: bytes) -> str:
    """Build a single RECORD entry line per PEP 376/427.

    Format: ``<name>,sha256=<base64-unpadded>,<size>``.
    """
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{member_name},sha256={b64},{len(data)}"


def _extract_so_files(wheel_path: Path) -> dict[str, bytes]:
    """Return a mapping of ``<module>.<py-tag>-<plat>.so`` -> bytes for every
    top-level ``.so`` member of the wheel.

    flash-attn subdir wheels put their compiled extension at the wheel root
    (sibling to ``<pkg>.dist-info/``), which is exactly what we want.
    """
    sos: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            if info.filename.endswith(".so") and "/" not in info.filename:
                sos[info.filename] = zf.read(info.filename)
    if not sos:
        raise RuntimeError(f"No top-level .so files found in {wheel_path}")
    return sos


def _py_tag() -> str:
    """e.g. cp312."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _platform_tag() -> str:
    """Match upstream flash-attn naming: linux_x86_64 / linux_aarch64."""
    machine = platform.machine()
    if machine in ("x86_64", "aarch64", "arm64"):
        return f"linux_{'aarch64' if machine == 'arm64' else machine}"
    raise RuntimeError(f"Unexpected machine {machine!r}; refine platform tagging.")


def _sanitize_build_tag(build_tag: str) -> str:
    """Filenames can't contain '-' in the build segment without confusing wheel
    parsers. Convert ``cu128torch2.9`` to a single token (already token-safe)."""
    if not re.fullmatch(r"[A-Za-z0-9._]+", build_tag):
        raise ValueError(f"build_tag must be alphanumeric/dot/underscore: {build_tag!r}")
    return build_tag


def _metadata_text(version: str) -> str:
    return (
        "Metadata-Version: 2.1\n"
        f"Name: odyssey-fused-kernels\n"
        f"Version: {version}\n"
        "Summary: Bundled flash-attention subdir kernels (dropout_layer_norm + fused_dense_lib) for Odyssey\n"
        "Home-page: https://github.com/odysseyml/flash-attention\n"
        "License: BSD-3-Clause\n"
        "Requires-Python: >=3.10\n"
        "\n"
        f"{DIST_INFO_DESCRIPTION}\n"
    )


def _wheel_text(py_tag: str, plat_tag: str) -> str:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: odyssey-bundle-script (1.0)\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {py_tag}-{py_tag}-{plat_tag}\n"
    )


def build_bundle(
    dropout_wheel: Path,
    fused_dense_wheel: Path,
    version: str,
    build_tag: str,
    output_dir: Path,
) -> Path:
    sos = _extract_so_files(dropout_wheel) | _extract_so_files(fused_dense_wheel)
    if len(sos) < 2:
        raise RuntimeError(
            f"Expected at least 2 distinct .so files across the two input wheels; got {list(sos)}"
        )

    py_tag = _py_tag()
    plat_tag = _platform_tag()
    safe_build_tag = _sanitize_build_tag(build_tag)
    wheel_filename = f"{PACKAGE_NAME}-{version}+{safe_build_tag}-{py_tag}-{py_tag}-{plat_tag}.whl"
    dist_info_dir = f"{PACKAGE_NAME}-{version}.dist-info"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / wheel_filename

    # Build RECORD as we add files. The RECORD entry for itself is special — it
    # contains no hash and no size (PEP 376).
    record_lines: list[str] = []
    members: list[tuple[str, bytes]] = []

    for so_name, so_bytes in sorted(sos.items()):
        members.append((so_name, so_bytes))
        record_lines.append(_record_entry(so_name, so_bytes))

    metadata = _metadata_text(version).encode("utf-8")
    members.append((f"{dist_info_dir}/METADATA", metadata))
    record_lines.append(_record_entry(f"{dist_info_dir}/METADATA", metadata))

    wheel_meta = _wheel_text(py_tag, plat_tag).encode("utf-8")
    members.append((f"{dist_info_dir}/WHEEL", wheel_meta))
    record_lines.append(_record_entry(f"{dist_info_dir}/WHEEL", wheel_meta))

    record_lines.append(f"{dist_info_dir}/RECORD,,")
    record_text = ("\n".join(record_lines) + "\n").encode("utf-8")
    members.append((f"{dist_info_dir}/RECORD", record_text))

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for name, data in members:
            zinfo = zipfile.ZipInfo(filename=name)
            zinfo.external_attr = 0o644 << 16
            out_zip.writestr(zinfo, data)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout-wheel", required=True, type=Path)
    parser.add_argument("--fused-dense-wheel", required=True, type=Path)
    parser.add_argument("--version", required=True, help="e.g. 0.1.0")
    parser.add_argument(
        "--build-tag",
        required=True,
        help="e.g. cu128torch2.9 — appears in the wheel filename as +<build-tag>",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    out = build_bundle(
        dropout_wheel=args.dropout_wheel,
        fused_dense_wheel=args.fused_dense_wheel,
        version=args.version,
        build_tag=args.build_tag,
        output_dir=args.output_dir,
    )
    print(f"Wrote {out}")
    print(f"  size: {out.stat().st_size:,} bytes")
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            print(f"  member: {info.filename}  ({info.file_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
