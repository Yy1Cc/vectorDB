#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace vectordb {

// TTLManager: TTL 过期管理
// 为向量设置过期时间，支持定期清理过期数据
class TTLManager {
public:
    TTLManager();

    // 设置 TTL（id 在 ttl_seconds 秒后过期）
    void SetTTL(uint64_t id, int64_t ttl_seconds);

    // 设置绝对过期时间戳（Unix 秒）
    void SetExpireAt(uint64_t id, int64_t expire_timestamp);

    // 检查是否过期
    bool IsExpired(uint64_t id) const;

    // 获取所有已过期 ID
    auto GetExpiredIDs() const -> std::vector<uint64_t>;

    // 删除 TTL 记录
    void Remove(uint64_t id);

    // 检查是否有 TTL 记录
    bool HasTTL(uint64_t id) const { return expire_times_.count(id) > 0; }

    // 获取过期时间戳
    int64_t GetExpireTime(uint64_t id) const;

    // 获取 TTL 记录数
    size_t Size() const { return expire_times_.size(); }

private:
    // id -> 过期时间戳（Unix 秒）
    std::map<uint64_t, int64_t> expire_times_;

    // 获取当前时间戳（Unix 秒）
    static int64_t CurrentTimestamp();
};

}  // namespace vectordb
