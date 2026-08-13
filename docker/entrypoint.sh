#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/opt/vectordb/third_party/installed/lib:${LD_LIBRARY_PATH:-}"

# 确保数据目录存在（volume 挂载点下的子目录不会自动创建）
mkdir -p /var/lib/vectordb/test/storage /var/lib/vectordb/test/wal /var/lib/vectordb/test/snap

if [[ $# -eq 0 ]]; then
  set -- vdb_server 1
fi

case "$1" in
  vdb_server)
    NODE_ID="${2:-1}"
    mkdir -p "/var/lib/vectordb/node-${NODE_ID}/storage" \
             "/var/lib/vectordb/node-${NODE_ID}/wal" \
             "/var/lib/vectordb/node-${NODE_ID}/snap"
    exec "/opt/vectordb/bin/$@"
    ;;
  vdb_server_master|vdb_server_proxy)
    exec "/opt/vectordb/bin/$@"
    ;;
  *)
    exec "$@"
    ;;
esac
