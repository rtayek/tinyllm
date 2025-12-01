echo off
set PYTHONPATH=src
del /q plots\* 2>nul
del /q checkpoints\* 2>nul
python -m llm.Main

