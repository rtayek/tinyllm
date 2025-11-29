#!/bin/sh
# tar the current directory, excluding __pycache__ dirs
tar --exclude='plots' --exclude='checkpoints' --exclude='*/__pycache__' --exclude='.git' --exclude='.pytest_cache' -czf ../tinyarchive.tar.gz .
