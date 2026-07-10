#include "index/fulltext_index.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "logger/logger.h"

namespace vectordb {

FullTextIndex::FullTextIndex() = default;

auto FullTextIndex::Tokenize(const std::string& text) const -> std::vector<std::string> {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

void FullTextIndex::UpdateAvgDocLength(const std::string& field) {
    auto it = doc_lengths_.find(field);
    if (it == doc_lengths_.end() || it->second.empty()) {
        avg_doc_length_[field] = 0.0;
        return;
    }
    int total = 0;
    for (const auto& [id, len] : it->second) {
        total += len;
    }
    avg_doc_length_[field] = static_cast<double>(total) / it->second.size();
}

void FullTextIndex::AddDocument(const std::string& field, const std::string& text, uint64_t id) {
    auto tokens = Tokenize(text);

    // 记录文档长度
    doc_lengths_[field][id] = static_cast<int>(tokens.size());

    // 统计词频
    std::map<std::string, int> tf_map;
    for (const auto& token : tokens) {
        tf_map[token]++;
    }

    // 写入倒排索引
    for (const auto& [term, tf] : tf_map) {
        inverted_index_[field][term][id] = tf;
    }

    // 更新文档计数
    doc_count_[field]++;

    // 更新平均文档长度
    UpdateAvgDocLength(field);

    global_logger->debug("FullTextIndex: added doc id={} to field='{}', tokens={}", id, field, tokens.size());
}

void FullTextIndex::RemoveDocument(const std::string& field, uint64_t id) {
    // 从倒排索引中删除
    auto field_it = inverted_index_.find(field);
    if (field_it != inverted_index_.end()) {
        for (auto term_it = field_it->second.begin(); term_it != field_it->second.end(); ) {
            auto doc_it = term_it->second.find(id);
            if (doc_it != term_it->second.end()) {
                term_it->second.erase(doc_it);
            }
            if (term_it->second.empty()) {
                term_it = field_it->second.erase(term_it);
            } else {
                ++term_it;
            }
        }
    }

    // 删除文档长度记录
    auto len_it = doc_lengths_.find(field);
    if (len_it != doc_lengths_.end()) {
        len_it->second.erase(id);
    }

    // 更新文档计数
    if (doc_count_[field] > 0) {
        doc_count_[field]--;
    }

    // 更新平均文档长度
    UpdateAvgDocLength(field);
}

auto FullTextIndex::Search(const std::string& field, const std::string& query, int k)
    -> std::pair<std::vector<uint64_t>, std::vector<float>> {
    auto tokens = Tokenize(query);
    auto field_it = inverted_index_.find(field);
    if (field_it == inverted_index_.end() || tokens.empty()) {
        return {{}, {}};
    }

    int N = doc_count_[field];
    double avgdl = avg_doc_length_[field];
    if (avgdl == 0.0) avgdl = 1.0; // 避免除零

    // 计算每个文档的 BM25 分数
    std::map<uint64_t, double> scores;

    for (const auto& term : tokens) {
        auto term_it = field_it->second.find(term);
        if (term_it == field_it->second.end()) continue;

        int n_t = static_cast<int>(term_it->second.size()); // 包含该词的文档数
        double idf = std::log(static_cast<double>(N - n_t + 0.5) / (n_t + 0.5) + 1.0);

        for (const auto& [doc_id, tf] : term_it->second) {
            int dl = doc_lengths_[field][doc_id];
            double denom = tf + k1_ * (1.0 - b_ + b_ * static_cast<double>(dl) / avgdl);
            double score = idf * (tf * (k1_ + 1.0)) / denom;
            scores[doc_id] += score;
        }
    }

    // 排序取 top-k
    std::vector<std::pair<double, uint64_t>> sorted_scores;
    for (const auto& [doc_id, score] : scores) {
        sorted_scores.push_back({score, doc_id});
    }
    int result_k = std::min(k, static_cast<int>(sorted_scores.size()));
    std::partial_sort(sorted_scores.begin(), sorted_scores.begin() + result_k,
                      sorted_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<uint64_t> result_ids;
    std::vector<float> result_scores;
    for (int i = 0; i < result_k; ++i) {
        result_ids.push_back(sorted_scores[i].second);
        result_scores.push_back(static_cast<float>(sorted_scores[i].first));
    }
    return {result_ids, result_scores};
}

auto FullTextIndex::GetScore(const std::string& field, const std::string& query, uint64_t id) -> float {
    auto [ids, scores] = Search(field, query, 1);
    // 这个方法不太高效，但用于简单场景
    // 更好的实现是直接计算单个文档的分数
    auto tokens = Tokenize(query);
    auto field_it = inverted_index_.find(field);
    if (field_it == inverted_index_.end() || tokens.empty()) return 0.0f;

    int N = doc_count_[field];
    double avgdl = avg_doc_length_[field];
    if (avgdl == 0.0) avgdl = 1.0;

    auto len_it = doc_lengths_[field].find(id);
    if (len_it == doc_lengths_[field].end()) return 0.0f;
    int dl = len_it->second;

    double total_score = 0.0;
    for (const auto& term : tokens) {
        auto term_it = field_it->second.find(term);
        if (term_it == field_it->second.end()) continue;

        auto doc_it = term_it->second.find(id);
        if (doc_it == term_it->second.end()) continue;

        int tf = doc_it->second;
        int n_t = static_cast<int>(term_it->second.size());
        double idf = std::log(static_cast<double>(N - n_t + 0.5) / (n_t + 0.5) + 1.0);
        double denom = tf + k1_ * (1.0 - b_ + b_ * static_cast<double>(dl) / avgdl);
        total_score += idf * (tf * (k1_ + 1.0)) / denom;
    }
    return static_cast<float>(total_score);
}

void FullTextIndex::SaveIndex(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        global_logger->error("FullTextIndex: failed to open file for saving: {}", path);
        return;
    }

    // 简单的文本序列化
    for (const auto& [field, term_map] : inverted_index_) {
        ofs << "F|" << field << "\n";
        for (const auto& [term, doc_map] : term_map) {
            for (const auto& [doc_id, tf] : doc_map) {
                ofs << "T|" << field << "|" << term << "|" << doc_id << "|" << tf << "\n";
            }
        }
    }
    for (const auto& [field, doc_len_map] : doc_lengths_) {
        for (const auto& [doc_id, len] : doc_len_map) {
            ofs << "L|" << field << "|" << doc_id << "|" << len << "\n";
        }
    }
    for (const auto& [field, count] : doc_count_) {
        ofs << "C|" << field << "|" << count << "\n";
    }
    ofs.close();
    global_logger->info("FullTextIndex: saved to {}", path);
}

void FullTextIndex::LoadIndex(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        global_logger->warn("FullTextIndex: file not found: {}", path);
        return;
    }

    inverted_index_.clear();
    doc_lengths_.clear();
    avg_doc_length_.clear();
    doc_count_.clear();

    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        char type;
        iss >> type;
        iss.ignore(1); // 跳过 '|'

        if (type == 'F') {
            // 字段标记，跳过（数据在 T 行中）
            continue;
        } else if (type == 'T') {
            std::string field, term;
            uint64_t doc_id;
            int tf;
            std::getline(iss, field, '|');
            std::getline(iss, term, '|');
            iss >> doc_id;
            iss.ignore(1);
            iss >> tf;
            inverted_index_[field][term][doc_id] = tf;
        } else if (type == 'L') {
            std::string field;
            uint64_t doc_id;
            int len;
            std::getline(iss, field, '|');
            iss >> doc_id;
            iss.ignore(1);
            iss >> len;
            doc_lengths_[field][doc_id] = len;
        } else if (type == 'C') {
            std::string field;
            int count;
            std::getline(iss, field, '|');
            iss >> count;
            doc_count_[field] = count;
        }
    }

    // 重新计算平均文档长度
    for (const auto& [field, _] : doc_count_) {
        UpdateAvgDocLength(field);
    }

    ifs.close();
    global_logger->info("FullTextIndex: loaded from {}", path);
}

}  // namespace vectordb
