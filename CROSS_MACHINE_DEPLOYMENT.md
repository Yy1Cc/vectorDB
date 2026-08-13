# VectorDB 跨机器部署说明

## 1. 文档目的

本文说明如何把当前 VectorDB 代码部署到多台机器上运行。目标是使用同一个 `vectordb:dev` 镜像，在不同机器上分别启动 `etcd`、`master`、`proxy` 和多个 `vdb_server`，验证真实网络环境下的 Raft 复制、Proxy 路由和节点故障恢复。

当前项目已经通过 Docker Compose 在单机上完成了多容器验证。单机 Compose 环境可以模拟多个进程和容器，但不能模拟真实的机器故障。跨机器部署时，需要把 Docker Compose 中的服务名替换成各机器可访问的 IP 或 DNS，并直接开放服务端口。

## 2. 组件职责

| 组件 | 主要职责 | 是否属于数据 Raft 副本组 |
|---|---|---|
| `etcd` | 保存节点、实例和分片等元数据 | 否，etcd 自身使用独立的 Raft |
| `master` | 访问 etcd，提供节点和分片配置接口，检查节点状态 | 否 |
| `proxy` | 接收客户端请求，按照分片配置转发到数据节点 | 否，无状态路由层 |
| `vdb_server` | 保存向量数据，提供 HTTP 服务，参与数据 Raft 复制 | 是 |

当前代码中的三个二进制都包含在同一个 `vectordb:dev` 镜像中，角色由启动命令决定：

```text
vdb_server 1          启动数据节点 1
vdb_server 2          启动数据节点 2
vdb_server 3          启动数据节点 3
vdb_server_master     启动 master
vdb_server_proxy      启动 proxy
```

## 3. 推荐的第一版跨机器拓扑

为了便于排查问题，第一轮建议使用 5 台机器：

```text
机器 A  10.0.0.10：etcd + master
机器 B  10.0.0.11：proxy
机器 C  10.0.0.12：vdb_server node-1
机器 D  10.0.0.13：vdb_server node-2
机器 E  10.0.0.14：vdb_server node-3
```

三个 `vdb_server` 组成一个 3 节点 Raft 副本组。`master` 通过 `etcd` 保存节点信息，`proxy` 通过 `master` 获取节点和分片信息。

后续可以将多个角色合并到同一台机器，例如在 3 台机器上分别运行控制面和数据节点，但当前代码的 etcd 仍是单实例，第一轮不建议同时引入多节点 etcd、高可用 master 和负载均衡器等额外变量。

## 4. 网络端口

跨机器部署前，确保主机防火墙和安全组允许以下访问：

| 端口 | 用途 | 访问方向 |
|---:|---|---|
| `2379` | etcd 客户端接口 | `master` → `etcd` |
| `6060` | master HTTP 接口 | `proxy` → `master`，运维端可选 |
| `6061` | proxy HTTP 接口 | 客户端 → `proxy` |
| `7781`、`7782`、`7783` | vdb_server HTTP 接口 | `proxy`、`master` → 对应 vdb_server |
| `8081`、`8082`、`8083` | NuRaft 节点间通信 | 三个 vdb_server 之间双向访问 |

需要区分两类 vdb_server 端口：

```text
8081/8082/8083：Raft 通信端口
7781/7782/7783：HTTP 请求端口
```

Raft 配置中的 `ENDPOINT` 使用 `808x` 端口；master 中注册的节点 `url` 使用 HTTP 的 `778x` 端口。

## 5. 分发镜像

先在构建机器生成镜像归档：

```bash
docker save vectordb:dev | gzip > vectordb-dev.tar.gz
```

将 `vectordb-dev.tar.gz` 复制到每台目标机器，在目标机器加载：

```bash
docker load < vectordb-dev.tar.gz
```

检查镜像：

```bash
docker images vectordb:dev
```

镜像中包含：

```text
/opt/vectordb/bin/vdb_server
/opt/vectordb/bin/vdb_server_master
/opt/vectordb/bin/vdb_server_proxy
/usr/local/bin/vectordb-entrypoint
```

配置文件和数据目录不应固化到镜像中。配置通过只读挂载提供，RocksDB、WAL、snapshot 和 Raft 日志通过数据卷持久化。

## 6. 配置文件

### 6.1 etcd 和 master 配置

机器 A 上的 `master_config`：

```json
{
    "MASTER_HOST": "0.0.0.0",
    "MASTER_PORT": 6060,
    "ETCD_ENDPOINTS": "http://10.0.0.10:2379"
}
```

`MASTER_HOST` 是监听地址，通常使用 `0.0.0.0`；`ETCD_ENDPOINTS` 必须填写 master 能访问的 etcd 地址，不能使用单机 Compose 中的 `etcd` 服务名。

### 6.2 proxy 配置

机器 B 上的 `proxy_config`：

```json
{
    "INSTANCE_ID": 1,
    "PROXY_ADDRESS": "0.0.0.0",
    "PROXY_PORT": 6061,
    "MASTER_HOST": "10.0.0.10",
    "MASTER_PORT": 6060,
    "READ_PATHS": ["/UserService/search"],
    "WRITE_PATHS": ["/UserService/upsert"]
}
```

`MASTER_HOST` 必须填写 master 的真实 IP 或可解析 DNS。

### 6.3 vdb_server 集群配置

三个数据节点可以挂载同一份包含全部节点信息的配置文件，但每个容器使用不同的启动参数：`vdb_server 1`、`vdb_server 2`、`vdb_server 3`。

配置中的节点地址改为真实 IP：

```json
{
    "CLUSTER_INFO": [
        {
            "RAFT": {
                "NODE_ID": 1,
                "ENDPOINT": "10.0.0.12:8081",
                "PORT": 8081
            },
            "ROCKS_DB_PATH": "/var/lib/vectordb/node-1/storage",
            "WAL_PATH": "/var/lib/vectordb/node-1/wal/wal.log",
            "SNAP_PATH": "/var/lib/vectordb/node-1/snap/",
            "ADDRESS": "0.0.0.0",
            "PORT": 7781
        },
        {
            "RAFT": {
                "NODE_ID": 2,
                "ENDPOINT": "10.0.0.13:8082",
                "PORT": 8082
            },
            "ROCKS_DB_PATH": "/var/lib/vectordb/node-2/storage",
            "WAL_PATH": "/var/lib/vectordb/node-2/wal/wal.log",
            "SNAP_PATH": "/var/lib/vectordb/node-2/snap/",
            "ADDRESS": "0.0.0.0",
            "PORT": 7782
        },
        {
            "RAFT": {
                "NODE_ID": 3,
                "ENDPOINT": "10.0.0.14:8083",
                "PORT": 8083
            },
            "ROCKS_DB_PATH": "/var/lib/vectordb/node-3/storage",
            "WAL_PATH": "/var/lib/vectordb/node-3/wal/wal.log",
            "SNAP_PATH": "/var/lib/vectordb/node-3/snap/",
            "ADDRESS": "0.0.0.0",
            "PORT": 7783
        }
    ]
}
```

每台数据机器必须挂载独立的数据目录。不能让三个节点共享同一个 RocksDB、WAL 或 snapshot 目录。

## 7. 启动命令

### 7.1 启动 etcd

在机器 A 执行：

```bash
docker run -d \
  --name vectordb-etcd \
  --restart unless-stopped \
  -p 2379:2379 \
  -v /srv/vectordb/etcd:/etcd-data \
  quay.io/coreos/etcd:v3.5.15 \
  etcd \
  --name=etcd \
  --advertise-client-urls=http://10.0.0.10:2379 \
  --listen-client-urls=http://0.0.0.0:2379 \
  --data-dir=/etcd-data
```

### 7.2 启动 master

在机器 A 准备 `/srv/vectordb/master_config` 后执行：

```bash
docker run -d \
  --name vectordb-master \
  --restart unless-stopped \
  -p 6060:6060 \
  -v /srv/vectordb/master_config:/opt/vectordb/master_config:ro \
  -v /srv/vectordb/master-data:/var/lib/vectordb \
  vectordb:dev \
  vdb_server_master
```

### 7.3 启动 proxy

在机器 B 准备 `/srv/vectordb/proxy_config` 后执行：

```bash
docker run -d \
  --name vectordb-proxy \
  --restart unless-stopped \
  -p 6061:6061 \
  -v /srv/vectordb/proxy_config:/opt/vectordb/proxy_config:ro \
  -v /srv/vectordb/proxy-data:/var/lib/vectordb \
  vectordb:dev \
  vdb_server_proxy
```

客户端访问：

```text
http://10.0.0.11:6061
```

### 7.4 启动三个 vdb_server

机器 C 启动 node-1：

```bash
docker run -d \
  --name vectordb-node-1 \
  --restart unless-stopped \
  -p 7781:7781 \
  -p 8081:8081 \
  -v /srv/vectordb/vectordb_config_cluster:/opt/vectordb/vectordb_config:ro \
  -v /srv/vectordb/node-1:/var/lib/vectordb \
  vectordb:dev \
  vdb_server 1
```

机器 D 启动 node-2：

```bash
docker run -d \
  --name vectordb-node-2 \
  --restart unless-stopped \
  -p 7782:7782 \
  -p 8082:8082 \
  -v /srv/vectordb/vectordb_config_cluster:/opt/vectordb/vectordb_config:ro \
  -v /srv/vectordb/node-2:/var/lib/vectordb \
  vectordb:dev \
  vdb_server 2
```

机器 E 启动 node-3：

```bash
docker run -d \
  --name vectordb-node-3 \
  --restart unless-stopped \
  -p 7783:7783 \
  -p 8083:8083 \
  -v /srv/vectordb/vectordb_config_cluster:/opt/vectordb/vectordb_config:ro \
  -v /srv/vectordb/node-3:/var/lib/vectordb \
  vectordb:dev \
  vdb_server 3
```

## 8. 集群初始化顺序

建议按照下面顺序操作：

1. 启动 etcd。
2. 启动 master，确认机器 A 的 `6060` 可访问。
3. 启动三个 `vdb_server`，分别确认 `7781`、`7782`、`7783` 和 `8081`、`8082`、`8083` 可访问。
4. 启动 proxy，确认 proxy 能访问 master。
5. 在 node-1 上调用 `AdminService/AddFollower`，将 node-2 和 node-3 加入 Raft。
6. 通过 master 的 `MasterService/AddNode` 注册三个数据节点的 HTTP URL。
7. 创建或更新实例的分片配置。
8. 通过 proxy 进行写入、查询和分片路由验证。

`AddFollower` 和 `AddNode` 使用的地址不同：

```text
AddFollower：使用 Raft 地址，例如 10.0.0.13:8082
AddNode：使用 HTTP 地址，例如 http://10.0.0.13:7782
```

假设 node-1 当前是 leader，可以在任意能访问机器 C 的主机上执行：

```bash
curl -X POST http://10.0.0.12:7781/AdminService/AddFollower \\
  -H 'Content-Type: application/json' \\
  -d '{"nodeId":2,"endpoint":"10.0.0.13:8082"}'

curl -X POST http://10.0.0.12:7781/AdminService/AddFollower \\
  -H 'Content-Type: application/json' \\
  -d '{"nodeId":3,"endpoint":"10.0.0.14:8083"}'
```

然后通过 master 注册数据节点。下面是三个节点的示例请求；如果部署代码对角色或状态有其他初始化约定，应以实际接口返回为准：

```bash
curl -X POST http://10.0.0.10:6060/MasterService/AddNode \\
  -H 'Content-Type: application/json' \\
  -d '{"instanceId":1,"nodeId":1,"url":"http://10.0.0.12:7781","role":0,"status":1}'

curl -X POST http://10.0.0.10:6060/MasterService/AddNode \\
  -H 'Content-Type: application/json' \\
  -d '{"instanceId":1,"nodeId":2,"url":"http://10.0.0.13:7782","role":1,"status":1}'

curl -X POST http://10.0.0.10:6060/MasterService/AddNode \\
  -H 'Content-Type: application/json' \\
  -d '{"instanceId":1,"nodeId":3,"url":"http://10.0.0.14:7783","role":1,"status":1}'
```

注册到 master 的节点对象至少需要包含 master 状态检查所使用的字段，例如：

```json
{
    "instanceId": 1,
    "nodeId": 1,
    "url": "http://10.0.0.12:7781",
    "role": 0,
    "status": 1
}
```

其中 `url` 是 master 和 proxy 访问数据节点 HTTP 接口的地址，不是 Raft endpoint。

## 9. 验证清单

### 服务连通性

从机器 B 验证 master：

```bash
curl http://10.0.0.10:6060/
```

从机器 B 验证三个数据节点的 HTTP 接口：

```bash
curl http://10.0.0.12:7781/AdminService/listCollections
curl http://10.0.0.13:7782/AdminService/listCollections
curl http://10.0.0.14:7783/AdminService/listCollections
```

### Raft 验证

检查以下结果：

```text
node-1、node-2、node-3 都能加入同一个 Raft 组
写入 leader 后三个节点都能 commit
停止 leader 后剩余节点能选出新 leader
向新 leader 写入成功并复制到另一个 follower
恢复旧 leader 后，旧 leader 能以 follower 身份重新加入
```

### Proxy 验证

客户端只访问 proxy 的 `6061`，不直接依赖某一个数据节点：

```text
客户端 → proxy:6061 → master:6060
                         ↓
              vdb_server:7781/7782/7783
```

## 10. 当前实现的限制

1. 当前 `docker-compose.yml` 使用 Docker 本机 bridge 网络，服务名解析只在同一台机器有效，不能直接作为跨机器编排方案。
2. 当前 Compose 只有一个 etcd 实例，不构成高可用 etcd 集群。
3. 当前 master 是单实例，master 宕机时 proxy 无法刷新元数据；数据节点已有的 Raft 组仍是独立的。
4. 当前 proxy 是单实例，生产环境需要额外的负载均衡或多个 proxy 实例。
5. 三个数据节点必须使用不同的持久化目录，不能共享宿主机目录或网络文件系统上的同一 RocksDB/WAL 路径。
6. 跨机器首次验证应优先使用固定 IP，确认通信稳定后再切换为 DNS。
7. 当前跨机器部署仍需要手动完成 Raft `AddFollower`、master `AddNode` 和分片配置初始化，不是完全自动化安装。

## 11. 与单机 Compose 的对应关系

单机 Compose 中的：

```text
etcd
master
proxy
vdb-node-1
vdb-node-2
vdb-node-3
```

跨机器后分别变成：

```text
10.0.0.10:2379       etcd
10.0.0.10:6060       master
10.0.0.11:6061       proxy
10.0.0.12:7781/8081  vdb-node-1
10.0.0.13:7782/8082  vdb-node-2
10.0.0.14:7783/8083  vdb-node-3
```

核心变化只有三点：

```text
Docker 服务名 → 真实 IP 或 DNS
Docker bridge 网络 → 主机网络和防火墙端口
单机 volume → 每台机器自己的持久化目录
```

## 12. 相关代码和配置

```text
Dockerfile
 docker-compose.yml
 docker/entrypoint.sh
 docker/config/master_config
 docker/config/proxy_config
 docker/config/vectordb_config_cluster
 tools/server/vdb_server.cpp
 tools/server/vdb_server_master.cpp
 tools/server/vdb_server_proxy.cpp
 src/cluster/raft_stuff.cpp
 src/httpserver/master_service_impl.cpp
 src/httpserver/proxy_service_impl.cpp
```
