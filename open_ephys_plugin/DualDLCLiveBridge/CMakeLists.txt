# plugin build file
cmake_minimum_required(VERSION 3.15)

include(../PluginRules.cmake)

add_sources(${PLUGIN_NAME}
            DualDLCLiveBridge.cpp
            DualDLCLiveBridge.h
            DualDLCLiveBridgeEditor.cpp
            DualDLCLiveBridgeEditor.h)

if(APPLE)
  set_target_properties(
    ${PLUGIN_NAME} PROPERTIES XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER
                              "org.open-ephys.plugin.DualDLCLiveBridge")
endif()

if(BUILD_TESTS)
  add_subdirectory(Tests)
endif()
