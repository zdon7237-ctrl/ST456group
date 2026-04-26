from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.error
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from novel_continuation.data_sources import SHERLOCK_SOURCES


DEFAULT_TIMEOUT = 30.0


def download_source(
    title: str,
    url: str,
    target_path: pathlib.Path,
    timeout: float = DEFAULT_TIMEOUT,
    skip_if_exists: bool = True,
) -> str:
    if skip_if_exists and target_path.exists():
        print(f"Skipping existing file for {title}: {target_path}")
        return "skipped"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "novel-continuation-downloader/0.1"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            target_path.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download {title} from {url} to {target_path}: {exc}"
        ) from exc

    print(f"Downloaded {title} -> {target_path}")
    return "downloaded"


def download_sources(
    output_dir: pathlib.Path,
    timeout: float = DEFAULT_TIMEOUT,
    skip_if_exists: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in SHERLOCK_SOURCES:
        target_path = output_dir / f"{source['book_id']}.txt"
        download_source(
            title=source["title"],
            url=source["url"],
            target_path=target_path,
            timeout=timeout,
            skip_if_exists=skip_if_exists,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Sherlock Holmes texts from Project Gutenberg."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory for the raw Project Gutenberg text files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Network timeout, in seconds, for each download request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_sources(
        output_dir=pathlib.Path(args.output_dir),
        timeout=args.timeout,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
