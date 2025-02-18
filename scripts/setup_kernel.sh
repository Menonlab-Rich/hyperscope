#! /usr/bin/env bash
# get the abs dirname of the current script
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_BIN_DIR="$1"
PROFILE="$("$1/ipython" profile locate)"
cp "$CWD/kernel_startup.py" "$PROFILE/startup/000_startup.py" & exit 0
