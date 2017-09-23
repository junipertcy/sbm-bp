set(SOURCE_FILES main.cpp blockmodel.cpp belief_propagation.cpp graph_utilities.cpp output_functions.cpp)
add_executable(bp ${SOURCE_FILES})

set_target_properties(bp PROPERTIES COMPILE_DEFINITIONS "OUTPUT_HISTORY=0")

if (Boost_FOUND)
    target_link_libraries(bp ${Boost_LIBRARIES})
endif (Boost_FOUND)
