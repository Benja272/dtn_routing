!/bin/bash
base_dir="$PWD"
cd comparation && python3 comparation.py && python3 graph_results.py && cd "$base_dir"