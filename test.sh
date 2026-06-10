#!/bin/sh
set -e

pyright
pytest --cov --cov-report=term-missing
