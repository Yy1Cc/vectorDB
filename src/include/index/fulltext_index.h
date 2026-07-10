#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vectordb {

// FullTextIndex: BM25 全文检索索引
// 对文本字段建立倒排索引，支持 BM25 相关性搜索
class FullTextIndex {
public:
    FullTextIndex();

    // 添加文档（field 字段的 text 内容，关联 id）
    void AddDocument(const std::string& field, const std::string& text, uint64_t id);

    // 删除文档
    void RemoveDocument(const std::string& field, uint64_t id);

    // BM25 搜索：返回 top-k 文档 ID 和分数
    auto Search(const std::string& field, const std::string& query, int k)
        -> std::pair<std::vector<uint64_t>, std::vector<float>>;

    // 获取单个文档的 BM25 分数（用于向量+全文混合搜索）
    auto GetScore(const std::string& field, const std::string& query, uint64_t id) -> float;

    // 获取字段中的文档数
    int GetDocCount(const std::string& field) const { return doc_count_.count(field) ? doc_count_.at(field) : 0; }

    // 序列化/反序列化
    void SaveIndex(const std::string& path);
    void LoadIndex(const std::string& path);

private:
    // 分词：按非字母数字字符分割，转小写
    auto Tokenize(const std::string& text) const -> std::vector<std::string>;

    // 重新计算平均文档长度
    void UpdateAvgDocLength(const std::string& field);

    // 倒排索引：field -> (term -> (doc_id -> tf))
    std::map<std::string, std::map<std::string, std::map<uint64_t, int>>> inverted_index_;

    // 文档长度：field -> (doc_id -> doc_length)
    std::map<std::string, std::map<uint64_t, int>> doc_lengths_;

    // 平均文档长度：field -> avgdl
    std::map<std::string, double> avg_doc_length_;

    // 文档总数：field -> N
    std::map<std::string, int> doc_count_;

    // BM25 参数
    static constexpr double k1_ = 1.2;
    static constexpr double b_ = 0.75;
};

}  // namespace vectordb
