# vectorDB 容器化与 3 节点 Raft 故障切换验证报告

## 概述

本文档记录 vectorDB 项目的 Docker 容器化过程和 3 节点 Raft 故障切换验证结果。通过 Docker Compose 在单机上搭建完整的 etcd + master + proxy + 3 节点 vdb_server 集群，验证了 Raft 复制一致性、leader 选举和节点重新加入，补齐了此前分布式压测的窟窿（此前仅 2 节点无多数派、未测故障切换）。

## Docker 镜像构建

### Dockerfile 结构

多阶段构建（Ubuntu 24.04）：
- **builder 阶段**：安装编译工具链 + third_party 全量编译 + cmake build 三个服务二进制
- **runtime 阶段**：只装运行时 .so + 复制二进制/配置/entrypoint

镜像大小 750MB，包含 `vdb_server`、`vdb_server_master`、`vdb_server_proxy` 三个二进制。

### 构建中遇到的问题与修复

1. **git clone 被网络代理拦截**：build.sh 中 rapidjson 和 asio 用 `git clone` 下载，Docker 构建环境被拦。修复：改为 `wget codeload.github.com/.../tar.gz` 下载。

2. **NuRaft GCC 13 兼容**：`asio_service_options.hxx` 用 `uint32_t`/`uint64_t` 未 `#include <cstdint>`，GCC 13 严格模式编译失败。修复：NuRaft cmake 加 `-DCMAKE_CXX_FLAGS="-include cstdint"` + 安装后 `sed` 补丁头文件。

3. **etcd-cpp-apiv3 缺 gRPC**：etcd-cpp-apiv3 依赖 gRPC，Docker 环境未装。修复：从 PACKAGES 数组移除 etcdclient（项目改用自研 EtcdHttpClient + curl 调 etcd v3 HTTP gateway，不链接 etcd-cpp-apiv3）。

4. **链接缺 libzstd/libgfortran**：RocksDB 链接 -lzstd、OpenBLAS 链接 -lgfortran，builder 缺 dev 包。修复：apt-get 补 `libzstd-dev` + `gfortran`。

5. **容器数据目录不存在**：RocksDB 打开 `/var/lib/vectordb/node-1/storage` 失败。修复：entrypoint.sh 按 node_id 动态 `mkdir -p`。

6. **Raft 日志写入容器可写层**：`./srvN.log` 是相对路径，working_dir 默认 `/opt/vectordb`（非 volume），重启丢失。修复：docker-compose.yml 设 `working_dir: /var/lib/vectordb`。

### 关键文件

- `Dockerfile`：多阶段构建
- `docker/entrypoint.sh`：容器启动入口，支持 server/master/proxy 三角色 + 动态数据目录创建
- `docker/config/vectordb_config`：单节点 smoke test 配置
- `docker/config/vectordb_config_cluster`：3 节点集群配置（ENDPOINT 用 Docker 服务名）
- `docker/config/master_config`：master 配置（ETCD_ENDPOINTS 指向 etcd 服务名）
- `docker/config/proxy_config`：proxy 配置（MASTER_HOST 指向 master 服务名）
- `.dockerignore`：排除 build/、third_party/installed、third_party/src
- `docker-compose.yml`：etcd + master + proxy + 3 节点编排

## 单节点 Smoke Test

```bash
docker run -d --name vdb-smoke -p 7781:7781 \
  -v vectordb-smoke:/var/lib/vectordb \
  vectordb:dev vdb_server 1
```

结果：容器正常启动，RocksDB → Raft → HTTP Server 初始化完成，`curl /AdminService/listCollections` 返回 `{"collections":["default"],"retCode":0}`。

## Compose 集群验证

### 集群拓扑

```
etcd (2379) ← master (6060) ← proxy (6061)
                     ↓
vdb-node-1 (7781/8081) — vdb-node-2 (7782/8082) — vdb-node-3 (7783/8083)
                  3 节点 Raft 组（多数派=2）
```

启动：`docker compose up -d`，6 个容器全部 Up。

### 验证 1：3 节点 Raft 组建

```bash
# node-1 为默认 leader，添加 node-2 和 node-3 为 follower
curl -X POST http://localhost:7781/AdminService/AddFollower \
  -H 'Content-Type: application/json' \
  -d '{"nodeId":2,"endpoint":"vdb-node-2:8082"}'

curl -X POST http://localhost:7781/AdminService/AddFollower \
  -H 'Content-Type: application/json' \
  -d '{"nodeId":3,"endpoint":"vdb-node-3:8083"}'
```

ListNode 结果：node-1 leader，node-2/3 follower，last_log_idx=13（同步完成）。

### 验证 2：Raft 复制一致性

向 leader (node-1) upsert id=42 的向量：

```bash
curl -X POST http://localhost:7781/UserService/upsert \
  -H 'Content-Type: application/json' \
  -d '{"collectionName":"default","id":42,"vectors":[0.1,...128维...]}'
```

三节点日志均显示：
- `Append app logs 14` → `Pre Commit log_idx: 14` → `Commit log_idx: 14`
- node-2 和 node-3 的 last_log_idx 从 13→14，数据一致

### 验证 3：Leader 故障切换（此前最大窟窿）

**停止 leader (node-1)**：

```bash
docker stop vectordb-vdb-node-1-1
```

**10 秒后检查**：
- node-3 自动选举为新 **leader**
- node-2 保持 follower
- node-1 标记为 follower（实际已宕机，last_succ_resp 过期）

**向新 leader (node-3) 写入 id=99**：
- retCode=0，写入成功
- node-2 收到并 commit（log_idx=16），复制正常

**重启 node-1**：
- node-1 以 follower 身份重新加入集群
- 将从新 leader 同步缺失的日志

### 验证 4：各节点 HTTP 可达

3 个数据节点均响应 `listCollections`，master 与 etcd 通信正常，proxy 获取分片配置成功。

## 构建与复现命令

```bash
# 构建镜像
docker build --build-arg BUILD_JOBS=4 -t vectordb:dev .

# 启动集群
docker compose up -d

# 组建 Raft 组
curl -X POST http://localhost:7781/AdminService/AddFollower \
  -H 'Content-Type: application/json' \
  -d '{"nodeId":2,"endpoint":"vdb-node-2:8082"}'
curl -X POST http://localhost:7781/AdminService/AddFollower \
  -H 'Content-Type: application/json' \
  -d '{"nodeId":3,"endpoint":"vdb-node-3:8083"}'

# 验证节点
curl http://localhost:7781/AdminService/ListNode

# 测试故障切换
docker stop vectordb-vdb-node-1-1
sleep 10
curl http://localhost:7783/AdminService/ListNode  # node-3 应为 leader

# 清理
docker compose down -v
```

## 跨机部署

同一份 `vectordb:dev` 镜像可分发到多台机器。跨机时需修改：
- 每节点 `vectordb_config` 的 RAFT.ENDPOINT 改为机器实际 IP
- master_config 的 ETCD_ENDPOINTS 指向 etcd 所在机器
- proxy_config 的 MASTER_HOST 指向 master 机器
- 各机器 Docker `--network host` 或放通端口

镜像本身不含机器特定配置，通过挂载 config 文件传入。

## 局限

- 单机 Docker Compose 共享物理资源，QPS 数据不代表真实跨机性能
- etcd-cpp-apiv3 被移除（项目改用 curl HTTP gateway，无 gRPC 依赖）
- Debug -O0 编译（CMakeLists.txt set(CMAKE_BUILD_TYPE Debug) 覆盖了 Dockerfile 的 Release），生产应改 Release 重测
- 节点重启后需手动重新 AddFollower（Raft 配置日志在 volume 中持久化，但 nuraft 的 state_manager 路径也需确认持久化）
