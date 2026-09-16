#!/usr/bin/env python3
"""Small boto3-based prefix transfer helper for Volt jobs."""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


DOWNLOAD_WORKERS = 32


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_prefix(uri: str, destination: Path,
                    expected_objects: int | None,
                    expected_bytes: int | None) -> None:
    bucket, prefix = parse_s3_uri(uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = boto3.client("s3", config=Config(max_pool_connections=64))
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(prefix):]
            if not relative or relative.startswith("../"):
                raise RuntimeError(f"Unsafe relative S3 key: {key!r}")
            items.append((key, relative, int(item["Size"])))
    objects = len(items)
    total_bytes = sum(item[2] for item in items)
    print(f"Inventory: {objects} objects / {total_bytes} bytes at {uri}", flush=True)
    if expected_objects is not None and objects != expected_objects:
        raise RuntimeError(
            f"Object-count mismatch: expected {expected_objects}, got {objects}")
    if expected_bytes is not None and total_bytes != expected_bytes:
        raise RuntimeError(
            f"Byte-count mismatch: expected {expected_bytes}, got {total_bytes}")

    completed = 0
    completed_lock = threading.Lock()
    transfer_config = TransferConfig(use_threads=False)

    def fetch(item: tuple[str, str, int]) -> None:
        nonlocal completed
        key, relative, _size = item
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(target), Config=transfer_config)
        with completed_lock:
            completed += 1
            if completed % 1000 == 0 or completed == objects:
                print(f"Downloaded {completed}/{objects} objects", flush=True)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        list(executor.map(fetch, items))
    print(f"Downloaded {objects} objects / {total_bytes} bytes from {uri}", flush=True)


def upload_directory(source: Path, uri: str) -> None:
    bucket, prefix = parse_s3_uri(uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    if not source.exists():
        print(f"Nothing to upload: {source} does not exist", flush=True)
        return
    client = boto3.client("s3")
    objects = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source).as_posix()
        client.upload_file(str(path), bucket, prefix + relative)
        objects += 1
    print(f"Uploaded {objects} objects from {source} to {uri}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download-prefix")
    download.add_argument("uri")
    download.add_argument("destination", type=Path)
    download.add_argument("--expected-objects", type=int)
    download.add_argument("--expected-bytes", type=int)
    upload = subparsers.add_parser("upload-directory")
    upload.add_argument("source", type=Path)
    upload.add_argument("uri")
    args = parser.parse_args()

    if args.command == "download-prefix":
        download_prefix(args.uri, args.destination,
                        args.expected_objects, args.expected_bytes)
    else:
        upload_directory(args.source, args.uri)


if __name__ == "__main__":
    main()
