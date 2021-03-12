#!/bin/bash
set -e

python3 entrypoint.py

exec "$@"
