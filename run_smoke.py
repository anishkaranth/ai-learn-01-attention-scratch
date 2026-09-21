#!/usr/bin/env python3
"""Print precomputed results/JSON.shot (full train smoke lived on ml-learn-13)."""
from __future__ import annotations
import json
from pathlib import Path
RESULTS = Path(__file__).resolve().parent / 'results'
def main():
    shot = RESULTS / 'JSON.shot'
    print(shot.read_text() if shot.exists() else json.dumps({'snapshot':'missing'}))
if __name__ == '__main__':
    main()
