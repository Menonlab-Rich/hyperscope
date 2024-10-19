#!/bin/bash

# Get the search directories
search_dirs=$(ld --verbose | grep SEARCH_DIR | sed -e 's/SEARCH_DIR("=\(.*\)");/\1/' | tr -s ' ' '\n' | sort | uniq)

# Function to search for libxcb files in a directory
search_libxcb() {
    local dir="$1"
    if [ -d "$dir" ]; then
        echo "Searching in $dir:"
        find "$dir" -name "libxcb*" -print
        echo
    fi
}

# Iterate through each search directory
while read -r dir; do
    search_libxcb "$dir"
done <<< "$search_dirs"
