#!/bin/sh

pyright
pytest --cov --cov-report=term-missing
