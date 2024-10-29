#!/bin/bash

src_dir="$1"
dst_dir="$2"
if [ -z "$src_dir" ] || [ -z "$dst_dir" ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi
mkdir -p "$dst_dir"
extract_archive() {
    archive="$1"
    dst="$2"
    tar xvf "$archive" -C "$dst"
}
export -f extract_archive
export dst_dir
find "$src_dir" -name "*.tar.gz" -print0 | xargs -0 -I {} -n 1 -P 64 bash -c 'extract_archive "{}" "$dst_dir"' _ "$dst_dir"
echo "All .tar.gz files have been extracted to $dst_dir."