/************************************************************************
Copyright 2017-2019 eBay Inc.
Author/Developer(s): Jung-Sang Ahn

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
**************************************************************************/

#pragma once

#include "cluster/in_memory_log_store.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include "libnuraft/nuraft.hxx"
#include "common/vector_cfg.h"
#include "database/vector_database.h"
#include "logger/logger.h"
namespace vectordb {

class InmemStateMgr : public nuraft::state_mgr {
 public:
  InmemStateMgr(int srv_id, const std::string &endpoint, VectorDatabase* vector_database)
      : my_id_(srv_id), my_endpoint_(endpoint), cur_log_store_(nuraft::cs_new<InmemLogStore>(vector_database)) {
    my_srv_config_ = nuraft::cs_new<nuraft::srv_config>(srv_id, endpoint);

    // Initial cluster config: contains only one server (myself).
    // 若磁盘上已有持久化的配置则优先恢复，否则节点重启后会"忘记"集群成员，
    // 表现为每次重启都必须手动 AddFollower。
    saved_config_ = nuraft::cs_new<nuraft::cluster_config>();
    saved_config_->get_servers().push_back(my_srv_config_);
    LoadFromDisk();
  }

  ~InmemStateMgr() override = default;

  auto load_config() -> nuraft::ptr<nuraft::cluster_config> override { return saved_config_; }

  void save_config(const nuraft::cluster_config &config) override {
    nuraft::ptr<nuraft::buffer> buf = config.serialize();
    saved_config_ = nuraft::cluster_config::deserialize(*buf);
    WriteBuffer(ConfigFilePath(), *buf);
  }

  void save_state(const nuraft::srv_state &state) override {
    // srv_state 含 term 与投票记录。不落盘会导致重启后 term 归零，
    // 节点可能接受本不该接受的投票，破坏选举安全性。
    nuraft::ptr<nuraft::buffer> buf = state.serialize();
    saved_state_ = nuraft::srv_state::deserialize(*buf);
    WriteBuffer(StateFilePath(), *buf);
  }

  auto read_state() -> nuraft::ptr<nuraft::srv_state> override { return saved_state_; }

  auto load_log_store() -> nuraft::ptr<nuraft::log_store> override {
    return std::static_pointer_cast<nuraft::log_store>(cur_log_store_);
  }

  auto server_id() -> nuraft::int32 override { return my_id_; }

  void system_exit(const int exit_code) override {}

  auto GetSrvConfig() const -> nuraft::ptr<nuraft::srv_config> { return my_srv_config_; }

 private:
  auto ConfigFilePath() const -> std::filesystem::path {
    return std::filesystem::path(Cfg::Instance().SnapPath()) /
           ("raft_config_" + std::to_string(my_id_) + ".bin");
  }

  auto StateFilePath() const -> std::filesystem::path {
    return std::filesystem::path(Cfg::Instance().SnapPath()) /
           ("raft_state_" + std::to_string(my_id_) + ".bin");
  }

  static void WriteBuffer(const std::filesystem::path& path, nuraft::buffer& buf) {
    // serialize() 结尾会 pos(0)，因此数据长度就是容量 size()，起点是 data_begin()。
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      global_logger->error("InmemStateMgr: cannot open {} for writing", path.string());
      return;
    }
    if (buf.size() > 0) {
      out.write(reinterpret_cast<const char*>(buf.data_begin()),
                static_cast<std::streamsize>(buf.size()));
    }
    out.flush();
  }

  static auto ReadBuffer(const std::filesystem::path& path) -> nuraft::ptr<nuraft::buffer> {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return nullptr;
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) return nullptr;
    nuraft::ptr<nuraft::buffer> buf = nuraft::buffer::alloc(bytes.size());
    buf->put(bytes.data(), bytes.size());
    buf->pos(0);   // deserialize 从当前 pos 开始读，必须回到 0
    return buf;
  }

  void LoadFromDisk() {
    if (auto buf = ReadBuffer(ConfigFilePath()); buf != nullptr) {
      try {
        saved_config_ = nuraft::cluster_config::deserialize(*buf);
        global_logger->info("InmemStateMgr: restored cluster config from {} ({} servers)",
                            ConfigFilePath().string(), saved_config_->get_servers().size());
      } catch (const std::exception& e) {
        global_logger->error("InmemStateMgr: failed to deserialize cluster config: {}", e.what());
      }
    }
    if (auto buf = ReadBuffer(StateFilePath()); buf != nullptr) {
      try {
        saved_state_ = nuraft::srv_state::deserialize(*buf);
        global_logger->info("InmemStateMgr: restored raft state (term={})", saved_state_->get_term());
      } catch (const std::exception& e) {
        global_logger->error("InmemStateMgr: failed to deserialize raft state: {}", e.what());
      }
    }
  }

  int my_id_;
  std::string my_endpoint_;
  nuraft::ptr<InmemLogStore> cur_log_store_;
  nuraft::ptr<nuraft::srv_config> my_srv_config_;
  nuraft::ptr<nuraft::cluster_config> saved_config_;
  nuraft::ptr<nuraft::srv_state> saved_state_;
};

}  // namespace vectordb
