#!/usr/bin/env bash
set -euo pipefail

# Build script to prepare frontend for Vercel static hosting
cp -R Frontend public
echo "Built frontend into ./public"
