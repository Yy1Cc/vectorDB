#pragma once
namespace vectordb {
#define LOGGER_NAME "GlobalLogger"

#define RESPONSE_VECTORS "vectors"
#define RESPONSE_DISTANCES "distances"

#define REQUEST_VECTORS "vectors"
#define REQUEST_K "k"
#define REQUEST_ID "id"
#define REQUEST_INDEX_TYPE "indexType"
#define INSTANCE_ID "instanceId"
#define NODE_ID "nodeId"

#define RESPONSE_RETCODE "retCode" // 添加宏定义
#define RESPONSE_RETCODE_SUCCESS 0
#define RESPONSE_RETCODE_ERROR (-1)

#define RESPONSE_ERROR_MSG "errorMsg" // 添加宏定义

#define RESPONSE_CONTENT_TYPE_JSON "application/json"
#define RESPONSE_CONTENT_TYPE_TEXT "text/plain"

#define INDEX_TYPE_FLAT "FLAT" // 添加宏定义
#define INDEX_TYPE_HNSW "HNSW" // 添加宏定义
#define INDEX_TYPE_SQ8 "SQ8"   // 8bit 标量量化索引
#define INDEX_TYPE_SQ4 "SQ4"   // 4bit 标量量化索引
#define INDEX_TYPE_IP_FLAT "IP_FLAT" // 内积(余弦)索引，ip2cos 预处理
#define INDEX_TYPE_IP_SQ8 "IP_SQ8"   // 内积(余弦)+SQ8 量化索引
#define INDEX_TYPE_LAYERED_FLAT "LAYERED_FLAT" // 分层存储 FLAT
#define INDEX_TYPE_LAYERED_SQ8 "LAYERED_SQ8"   // 分层存储 SQ8
#define REQUEST_FULLTEXT "fulltext"   // 全文搜索参数
#define REQUEST_TTL "ttl"             // TTL 过期参数（秒）

#define REQUEST_OPERATION_TYPE "operationType" // 批量操作类型字段
#define OPERATION_TYPE_BATCH_UPSERT "batch_upsert" // 批量插入操作类型
#define REQUEST_ITEMS "items" // 批量操作的数据项数组

#define REQUEST_COLLECTION_NAME "collectionName" // Collection 名称字段
#define DEFAULT_COLLECTION_NAME "default" // 默认 Collection 名称

// 其他字符串常量...
}  // namespace vectordb