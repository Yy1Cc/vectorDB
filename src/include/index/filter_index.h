#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <memory> // 包含 <memory> 以使用 std::shared_ptr
#include "database/scalar_storage.h"
#include "roaring/roaring.h"

namespace vectordb {

class FilterIndex {
public:
    enum class Operation {
        EQUAL,
        NOT_EQUAL,
        GREATER_THAN,    // >
        LESS_THAN,       // <
        GREATER_EQUAL,   // >=
        LESS_EQUAL       // <=
    };

    FilterIndex();

    // int 字段
    void AddIntFieldFilter(const std::string& fieldname, int64_t value, uint64_t id);
    void UpdateIntFieldFilter(const std::string& fieldname, int64_t* old_value, int64_t new_value, uint64_t id);
    void GetIntFieldFilterBitmap(const std::string& fieldname, Operation op, int64_t value, roaring_bitmap_t* result_bitmap);

    // 字符串字段
    void AddStringFieldFilter(const std::string& fieldname, const std::string& value, uint64_t id);
    void UpdateStringFieldFilter(const std::string& fieldname, const std::string* old_value, const std::string& new_value, uint64_t id);
    void GetStringFieldFilterBitmap(const std::string& fieldname, Operation op, const std::string& value, roaring_bitmap_t* result_bitmap);

    auto SerializeIntFieldFilter() -> std::string;
    void DeserializeIntFieldFilter(const std::string& serialized_data);
    void SaveIndex(const std::string& path);
    void LoadIndex(const std::string& path);

private:
    std::map<std::string, std::map<int64_t, roaring_bitmap_t*>> int_field_filter_;
    std::map<std::string, std::map<std::string, roaring_bitmap_t*>> string_field_filter_;
};

}  // namespace vectordb