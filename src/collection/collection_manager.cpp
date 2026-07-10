#include "collection/collection_manager.h"
#include "logger/logger.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace vectordb {

bool CollectionManager::CreateCollection(const std::string& name, int dim, int num_data,
                                          IndexFactory::MetricType metric) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (collections_.find(name) != collections_.end()) {
        global_logger->warn("Collection '{}' already exists", name);
        return false;
    }
    if (dim <= 0) {
        global_logger->error("CreateCollection '{}': invalid dimension {}", name, dim);
        return false;
    }

    auto coll = std::make_unique<Collection>();
    coll->meta.name = name;
    coll->meta.dimension = dim;
    coll->meta.num_data = num_data;
    coll->meta.metric = metric;

    // 生成创建时间戳
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&now_time), "%Y-%m-%dT%H:%M:%SZ");
    coll->meta.created_at = oss.str();

    // 初始化全部 9 种索引
    coll->InitAllIndices();

    collections_[name] = std::move(coll);
    global_logger->info("Created collection '{}' dim={} num_data={}", name, dim, num_data);
    return true;
}

bool CollectionManager::DropCollection(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = collections_.find(name);
    if (it == collections_.end()) {
        global_logger->warn("DropCollection '{}': not found", name);
        return false;
    }
    // 不允许删除 default collection（保持向后兼容）
    if (name == "default") {
        global_logger->error("Cannot drop default collection");
        return false;
    }
    collections_.erase(it);
    global_logger->info("Dropped collection '{}'", name);
    return true;
}

Collection* CollectionManager::GetCollection(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<std::string> CollectionManager::ListCollections() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(collections_.size());
    for (const auto& [name, _] : collections_) {
        names.push_back(name);
    }
    return names;
}

bool CollectionManager::HasCollection(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return collections_.find(name) != collections_.end();
}

bool CollectionManager::GetCollectionMeta(const std::string& name, CollectionMeta& meta) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = collections_.find(name);
    if (it == collections_.end()) {
        return false;
    }
    meta = it->second->meta;
    return true;
}

void CollectionManager::SaveToDisk(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& alloc = doc.GetAllocator();

    rapidjson::Value collections_array(rapidjson::kArrayType);
    for (const auto& [name, coll] : collections_) {
        rapidjson::Value obj(rapidjson::kObjectType);
        obj.AddMember("name", rapidjson::Value(coll->meta.name.c_str(), alloc), alloc);
        obj.AddMember("dimension", coll->meta.dimension, alloc);
        obj.AddMember("num_data", coll->meta.num_data, alloc);
        obj.AddMember("metric",
                      rapidjson::Value(coll->meta.metric == IndexFactory::MetricType::IP ? "IP" : "L2", alloc),
                      alloc);
        obj.AddMember("created_at", rapidjson::Value(coll->meta.created_at.c_str(), alloc), alloc);
        collections_array.PushBack(obj, alloc);
    }
    doc.AddMember("collections", collections_array, alloc);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        global_logger->error("SaveToDisk: cannot open file {}", path);
        return;
    }
    ofs << buffer.GetString();
    ofs.close();
    global_logger->info("Saved {} collections to {}", collections_.size(), path);
}

void CollectionManager::LoadFromDisk(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        global_logger->info("LoadFromDisk: file {} not found, no collections to load", path);
        return;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    rapidjson::Document doc;
    doc.Parse(content.c_str());
    if (!doc.IsObject() || !doc.HasMember("collections") || !doc["collections"].IsArray()) {
        global_logger->error("LoadFromDisk: invalid JSON format in {}", path);
        return;
    }

    const auto& arr = doc["collections"].GetArray();
    for (rapidjson::SizeType i = 0; i < arr.Size(); ++i) {
        const auto& obj = arr[i];
        if (!obj.HasMember("name") || !obj.HasMember("dimension")) {
            continue;
        }
        std::string name = obj["name"].GetString();
        int dim = obj["dimension"].GetInt();
        int num_data = obj.HasMember("num_data") ? obj["num_data"].GetInt() : 1000000;
        IndexFactory::MetricType metric = IndexFactory::MetricType::L2;
        if (obj.HasMember("metric") && std::string(obj["metric"].GetString()) == "IP") {
            metric = IndexFactory::MetricType::IP;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (collections_.find(name) != collections_.end()) {
            global_logger->warn("LoadFromDisk: collection '{}' already exists, skipping", name);
            continue;
        }
        auto coll = std::make_unique<Collection>();
        coll->meta.name = name;
        coll->meta.dimension = dim;
        coll->meta.num_data = num_data;
        coll->meta.metric = metric;
        if (obj.HasMember("created_at")) {
            coll->meta.created_at = obj["created_at"].GetString();
        }
        coll->InitAllIndices();
        collections_[name] = std::move(coll);
        global_logger->info("Loaded collection '{}' dim={} from disk", name, dim);
    }
}

}  // namespace vectordb
