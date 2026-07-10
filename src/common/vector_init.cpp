#include "common/vector_init.h"
#include "common/master_cfg.h"
#include "common/proxy_cfg.h"
#include "common/vector_cfg.h"
#include "index/index_factory.h"
#include "logger/logger.h"
#include "database/persistence.h"
namespace vectordb {
void VdbServerInit(int node_id) {
  auto cfg_path = GetCfgPath("vectordb_config");
  Cfg::SetCfg(cfg_path,node_id);
  InitGlobalLogger(Cfg::Instance().GlogName());
  SetLogLevel(Cfg::Instance().GlogLevel());
  auto &indexfactory = IndexFactory::Instance();
  int dim = Cfg::Instance().Dim();
  int num_data = Cfg::Instance().NumData();
  indexfactory.Init(IndexFactory::IndexType::FLAT, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::HNSW, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::FILTER, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::SQ8, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::SQ4, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::IP_FLAT, dim, num_data, IndexFactory::MetricType::IP);
  indexfactory.Init(IndexFactory::IndexType::IP_SQ8, dim, num_data, IndexFactory::MetricType::IP);
  indexfactory.Init(IndexFactory::IndexType::LAYERED_FLAT, dim, num_data);
  indexfactory.Init(IndexFactory::IndexType::LAYERED_SQ8, dim, num_data);
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