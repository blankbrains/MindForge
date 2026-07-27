#!/usr/bin/env bash
# QA 生成启动脚本
# 使用方法：bash scripts/run_qa_gen.sh [--all | --domain <name>]
export PYTHONIOENCODING=utf-8
exec python "$(dirname "$0")/generate_qa_dataset.py" "$@"
