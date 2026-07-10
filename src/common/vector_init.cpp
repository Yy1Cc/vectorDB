#include "common/vector_init.h"
#include "common/master_cfg.h"
#include "common/proxy_cfg.h"
#include "common/vector_cfg.h"
#include "common/constants.h"
#include "collection/collection_manager.h"
#include "index/index_factory.h"
#include "logger/logger.h"
#include "database/persistence.h"
namespace vectordb {
void VdbServerInit(int node_id) {
  auto cfg_path = GetCfgPath("vectordb_config");
  Cfg::SetCfg(cfg_path,node_id);
  InitGlobalLogger(Cfg::Instance().GlogName());
  SetLogLevel(Cfg::Instance().GlogLevel());

  // 从磁盘加载已持久化的 Collection 元数据
  std::string rocks_db_path = Cfg::Instance().RocksDbPath();
  std::string meta_path = rocks_db_path + "/collections_meta.json";
  CollectionManager::Instance().LoadFromDisk(meta_path);

  // 如果没有任何 Collection（首次启动），从 config 创建 default Collection
  if (!CollectionManager::Instance().HasCollection(DEFAULT_COLLECTION_NAME)) {
    int dim = Cfg::Instance().Dim();
    int num_data = Cfg::Instance().NumData();
    CollectionManager::Instance().CreateCollection(DEFAULT_COLLECTION_NAME, dim, num_data);
    CollectionManager::Instance().SaveToDisk(meta_path);
  }
}


void ProxyServerInit() {
  auto cfg_path = GetCfgPath("proxy_config");
  ProxyCfg::SetCfg(cfg_path);
  InitGlobalLogger(ProxyCfg::Instance().GlogName());
  SetLogLevel(ProxyCfg::Instance().GlogLevel());
}

void MasterServerInit() {
  auto cfg_path = GetCfgPath("master_config");
  MasterCfg::SetCfg(cfg_path);
  InitGlobalLogger(MasterCfg::Instance().GlogName());
  SetLogLevel(MasterCfg::Instance().GlogLevel());
}

}  // namespace vectordb