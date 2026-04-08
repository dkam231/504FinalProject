import argparse
import asyncio
import json
from pathlib import Path

import httpx


def load_image_records(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("images", [])


async def download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    out_path: Path,
    retries: int,
):
    async with semaphore:
        if out_path.exists():
            return "skipped"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_err = None
        for _ in range(retries + 1):
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                return "downloaded"
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(1.0)

        return f"failed: {last_err}"


async def download_from_jsons(
    json_paths,
    out_dir: Path,
    max_workers: int,
    timeout_s: float,
    retries: int,
):
    all_records = []
    for json_path in json_paths:
        images = load_image_records(json_path)
        all_records.extend(images)

    # Deduplicate by file name (same file can appear across sources)
    by_name = {}
    for rec in all_records:
        name = rec.get("file_name")
        url = rec.get("coco_url")
        if name and url:
            by_name[name] = url

    semaphore = asyncio.Semaphore(max_workers)
    timeout = httpx.Timeout(timeout_s)
    limits = httpx.Limits(max_connections=max_workers * 2, max_keepalive_connections=max_workers)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
        tasks = []
        for file_name, url in by_name.items():
            out_path = out_dir / file_name
            tasks.append(
                download_one(
                    client=client,
                    semaphore=semaphore,
                    url=url,
                    out_path=out_path,
                    retries=retries,
                )
            )

        downloaded = 0
        skipped = 0
        failed = 0

        total = len(tasks)
        for i, result in enumerate(asyncio.as_completed(tasks), start=1):
            status = await result
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"[warn] {status}")

            if i % 200 == 0 or i == total:
                print(
                    f"Progress: {i}/{total} | downloaded={downloaded} skipped={skipped} failed={failed}"
                )

    print("\nDone.")
    print(f"Total unique files: {len(by_name)}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Failed: {failed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download images referenced in FathomNet segmentation JSON files."
    )
    parser.add_argument(
        "--json-paths",
        nargs="+",
        required=True,
        help="One or more COCO JSON paths (e.g., fathomnet_segmentations/train.json test.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="images",
        help="Output image directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per image before marking as failed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_paths = [Path(p).resolve() for p in args.json_paths]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in json_paths:
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")

    asyncio.run(
        download_from_jsons(
            json_paths=json_paths,
            out_dir=out_dir,
            max_workers=args.max_workers,
            timeout_s=args.timeout,
            retries=args.retries,
        )
    )


if __name__ == "__main__":
    main()
