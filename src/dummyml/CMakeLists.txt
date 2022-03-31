include_directories(${dummyml_SOURCE_DIR}/include/dummyml)

set(
    dummyml_SOURCE_FILES
    ${dummyml_SOURCE_FILES}
    ${dummyml_SOURCE_FILES}/src/dummyml/dummy_classifier.cpp
    ${dummyml_SOURCE_FILES}/src/dummyml/utils.cpp
    )