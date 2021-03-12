#!/bin/bash
set -e

python3 COMPSs-obstacle-detection/entrypoint/entrypoint.py

exec "$@"
