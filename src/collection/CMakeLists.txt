add_library(
        vectorDB_collection
        OBJECT
        collection_manager.cpp
)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:vectorDB_collection>
        PARENT_SCOPE)
