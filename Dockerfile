# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    perl \
    pkg-config \
    python3 \
    patch \
    tar \
    unzip \
    wget \
    autoconf \
    automake \
    libtool \
    nasm \
    yasm \
    libssl-dev \
    libcurl4-openssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libsnappy-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libopenblas-dev \
    libzstd-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# third_party/build.sh 固定安装到 /src/third_party/installed，随后根 CMake 使用同一目录构建。
# build.sh 已改为用 wget codeload tar.gz 替代 git clone，避免 Docker 构建环境网络代理拦截。
RUN bash third_party/build.sh --j "${BUILD_JOBS}"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "${BUILD_JOBS}" \
       --target vdb_server vdb_server_master vdb_server_proxy

FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libbz2-1.0 \
    libcurl4 \
    libgfortran5 \
    libgomp1 \
    liblz4-1 \
    libsnappy1v5 \
    libssl3 \
    libstdc++6 \
    libz1 \
    libzstd1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vectordb
ENV VECTORDB_CODE_BASE=/opt/vectordb

COPY --from=builder /src/build/bin/vdb_server /opt/vectordb/bin/vdb_server
COPY --from=builder /src/build/bin/vdb_server_master /opt/vectordb/bin/vdb_server_master
COPY --from=builder /src/build/bin/vdb_server_proxy /opt/vectordb/bin/vdb_server_proxy
COPY --from=builder /src/third_party/installed/lib/libcurl.so.4 /opt/vectordb/third_party/installed/lib/libcurl.so.4
COPY docker/config/vectordb_config /opt/vectordb/vectordb_config
COPY docker/config/master_config /opt/vectordb/master_config
COPY docker/config/proxy_config /opt/vectordb/proxy_config
COPY docker/entrypoint.sh /usr/local/bin/vectordb-entrypoint

RUN ln -s libcurl.so.4 /opt/vectordb/third_party/installed/lib/libcurl.so \
    && chmod +x /usr/local/bin/vectordb-entrypoint \
    && mkdir -p /var/lib/vectordb

VOLUME ["/var/lib/vectordb"]
ENTRYPOINT ["/usr/local/bin/vectordb-entrypoint"]
CMD ["vdb_server", "1"]
