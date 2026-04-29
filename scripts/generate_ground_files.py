"""
Generate a deterministic FILES_GROUND list from Kitware Data.

Default behavior:
- Source folder: 56f581ce8d777f753209ca43
- Include only video files
- Keep files strictly below 200 MB
- Select first 25 after deterministic sort
- Print as Python block:
    FILES_GROUND = [
        ("<id>", "<filename>", <size>),
        ...
    ]

Notes:
- Uses item IDs in output by default (compatible with /api/v1/item/{id}/download).
- Use --id-kind file to output file IDs instead.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request


DEFAULT_FOLDER_ID = "56f581ce8d777f753209ca43"
DEFAULT_MAX_MB = 200
VIDEO_EXTENSIONS = (".mpg", ".mpeg", ".mp4", ".avi", ".mov", ".mkv")
ITEMS_URL = "https://data.kitware.com/api/v1/item"
ITEM_FILES_URL = "https://data.kitware.com/api/v1/item/{}/files"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FILES_GROUND from Kitware folder contents")
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID, help="Kitware folder ID")
    parser.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB, help="Maximum file size in MB (exclusive)")
    parser.add_argument(
        "--id-kind",
        choices=("item", "file"),
        default="item",
        help="Whether first tuple element is item ID or file ID",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save selected tuples as JSON",
    )
    return parser.parse_args()


def read_json(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def list_items(folder_id):
    all_items = []
    limit = 100
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "folderId": folder_id,
                "limit": limit,
                "offset": offset,
                "sort": "name",
                "sortdir": 1,
            }
        )
        chunk = read_json(f"{ITEMS_URL}?{query}")
        if not chunk:
            break
        all_items.extend(chunk)
        offset += len(chunk)
    return all_items


def list_files_for_item(item_id):
    query = urllib.parse.urlencode(
        {
            "limit": 200,
            "offset": 0,
            "sort": "name",
            "sortdir": 1,
        }
    )
    return read_json(f"{ITEM_FILES_URL.format(item_id)}?{query}")


def is_video(name):
    return name.lower().endswith(VIDEO_EXTENSIONS)


def build_rows(folder_id, max_bytes, id_kind):
    rows = []
    items = list_items(folder_id)
    for item in items:
        item_id = item["_id"]
        files = list_files_for_item(item_id)
        for file_obj in files:
            filename = file_obj.get("name", "")
            size = int(file_obj.get("size", 0))
            if not is_video(filename):
                continue
            if size >= max_bytes:
                continue
            source_id = item_id if id_kind == "item" else file_obj["_id"]
            rows.append((source_id, filename, size))
    return rows


def format_python_block(rows):
    lines = ["FILES_GROUND = ["]
    for source_id, filename, size in rows:
        lines.append(f'    ("{source_id}", "{filename}", {size}),')
    lines.append("]")
    return "\n".join(lines)


def main():
    args = parse_args()
    max_bytes = args.max_mb * 1024 * 1024

    rows = build_rows(args.folder_id, max_bytes, args.id_kind)
    rows.sort(key=lambda x: (x[1].lower(), x[2], x[0]))
    selected = rows

    print(f"# folder_id={args.folder_id}")
    print(f"# max_mb={args.max_mb} (exclusive), id_kind={args.id_kind}, selected={len(selected)}")
    print(format_python_block(selected))

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        payload = [
            {"source_id": source_id, "filename": filename, "size": size}
            for source_id, filename, size in selected
        ]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
