#!/usr/bin/env bash

source ./.pypath
# source ./.env
export META_DATA_PATH='datasets/bagnet_gnn'
export PYTHONHASHSEED=0

exec python $@