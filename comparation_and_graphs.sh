#!/bin/bash
base_dir="$PWD"
cd comparation && python3 comparations.py && cd results && . run_sims.sh && cd .. && python3 graph_results.py && cd "$base_dir"