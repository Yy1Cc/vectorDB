#include "database/ttl_manager.h"
#include <chrono>
#include "logger/logger.h"

namespace vectordb {

TTLManager::TTLManager() = default;

int64_t TTLManager::CurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void TTLManager::SetTTL(uint64_t id, int64_t ttl_seconds) {
    int64_t expire = CurrentTimestamp() + ttl_seconds;
    expire_times_[id] = expire;
    global_logger->debug("TTLManager: set TTL for id={}, expire_at={}", id, expire);
}

void TTLManager::SetExpireAt(uint64_t id, int64_t expire_timestamp) {
    expire_times_[id] = expire_timestamp;
    global_logger->debug("TTLManager: set expire_at for id={}, expire_at={}", id, expire_timestamp);
}

bool TTLManager::IsExpired(uint64_t id) const {
    auto it = expire_times_.find(id);
    if (it == expire_times_.end()) return false;
    return CurrentTimestamp() >= it->second;
}

auto TTLManager::GetExpiredIDs() const -> std::vector<uint64_t> {
    int64_t now = CurrentTimestamp();
    std::vector<uint64_t> expired;
    for (const auto& [id, expire] : expire_times_) {
        if (now >= expire) {
            expired.push_back(id);
        }
    }
    return expired;
}

void TTLManager::Remove(uint64_t id) {
    expire_times_.erase(id);
}

int64_t TTLManager::GetExpireTime(uint64_t id) const {
    auto it = expire_times_.find(id);
    if (it == expire_times_.end()) return -1;
    return it->second;
}

}  // namespace vectordb
