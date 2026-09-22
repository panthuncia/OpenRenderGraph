cmake_minimum_required(VERSION 3.16)

project(OpenRenderGraph VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)

option(OPENRENDERGRAPH_ENABLE_SUBMODULE_FALLBACK "Allow fallback to in-tree BasicRHI when package is unavailable" ON)
option(OPENRENDERGRAPH_FORWARD_BASICRHI_DEP_OPTIONS "Forward BasicRHI PIX/Streamline options when adding BasicRHI subdirectory" ON)
option(OPENRENDERGRAPH_ENABLE_DEBUG_UI "Build the optional ImGui/ImPlot/Slang inspector" OFF)
option(OPENRENDERGRAPH_ENABLE_D3D11_INTEROP "Build the optional Windows D3D11/D3D12 interop library" OFF)
option(OPENRENDERGRAPH_BUILD_TESTS "Build OpenRenderGraph tests" OFF)

set(OPENRENDERGRAPH_BASICRHI_ENABLE_STREAMLINE "" CACHE STRING "Override BASICRHI_ENABLE_STREAMLINE for submodule fallback (ON/OFF). Empty keeps BasicRHI default.")
set(OPENRENDERGRAPH_BASICRHI_ENABLE_PIX "" CACHE STRING "Override BASICRHI_ENABLE_PIX for submodule fallback (ON/OFF). Empty keeps BasicRHI default.")
set(OPENRENDERGRAPH_BASICRHI_STREAMLINE_HEADERS_DIR "" CACHE PATH "Override BASICRHI_STREAMLINE_HEADERS_DIR for submodule fallback.")
set(OPENRENDERGRAPH_BASICRHI_PIX_HEADERS_DIR "" CACHE PATH "Override BASICRHI_PIX_HEADERS_DIR for submodule fallback.")

find_package(BasicRHI CONFIG QUIET)
if(TARGET BasicRHI AND NOT TARGET BasicRHI::BasicRHI)
    add_library(BasicRHI::BasicRHI ALIAS BasicRHI)
endif()

if(NOT TARGET BasicRHI::BasicRHI AND OPENRENDERGRAPH_ENABLE_SUBMODULE_FALLBACK)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../BasicRHI/CMakeLists.txt")
        if(OPENRENDERGRAPH_FORWARD_BASICRHI_DEP_OPTIONS)
            if(NOT OPENRENDERGRAPH_BASICRHI_ENABLE_STREAMLINE STREQUAL "")
                set(BASICRHI_ENABLE_STREAMLINE "${OPENRENDERGRAPH_BASICRHI_ENABLE_STREAMLINE}" CACHE BOOL "" FORCE)
            endif()
            if(NOT OPENRENDERGRAPH_BASICRHI_ENABLE_PIX STREQUAL "")
                set(BASICRHI_ENABLE_PIX "${OPENRENDERGRAPH_BASICRHI_ENABLE_PIX}" CACHE BOOL "" FORCE)
            endif()
            if(NOT OPENRENDERGRAPH_BASICRHI_STREAMLINE_HEADERS_DIR STREQUAL "")
                set(BASICRHI_STREAMLINE_HEADERS_DIR "${OPENRENDERGRAPH_BASICRHI_STREAMLINE_HEADERS_DIR}" CACHE PATH "" FORCE)
            endif()
            if(NOT OPENRENDERGRAPH_BASICRHI_PIX_HEADERS_DIR STREQUAL "")
                set(BASICRHI_PIX_HEADERS_DIR "${OPENRENDERGRAPH_BASICRHI_PIX_HEADERS_DIR}" CACHE PATH "" FORCE)
            endif()
        endif()
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../BasicRHI" "${CMAKE_BINARY_DIR}/_deps/BasicRHI")
    endif()
endif()

if(TARGET BasicRHI AND NOT TARGET BasicRHI::BasicRHI)
    add_library(BasicRHI::BasicRHI ALIAS BasicRHI)
endif()

if(NOT TARGET BasicRHI::BasicRHI)
    message(FATAL_ERROR "OpenRenderGraph requires BasicRHI::BasicRHI. Install BasicRHI package or provide BasicRHI as submodule.")
endif()

find_package(BasicTelemetry CONFIG QUIET)
if(NOT TARGET BasicTelemetry::Core AND OPENRENDERGRAPH_ENABLE_SUBMODULE_FALLBACK)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../BasicTelemetry/CMakeLists.txt")
        set(BASICTELEMETRY_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BASICTELEMETRY_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        add_subdirectory(
            "${CMAKE_CURRENT_LIST_DIR}/../BasicTelemetry"
            "${CMAKE_BINARY_DIR}/_deps/BasicTelemetry")
    endif()
endif()
if(NOT TARGET BasicTelemetry::Core OR NOT TARGET BasicTelemetry::Tracy)
    message(FATAL_ERROR "OpenRenderGraph requires BasicTelemetry::Core and BasicTelemetry::Tracy.")
endif()

find_package(spdlog CONFIG REQUIRED)
# Must match the spdlog flavor the host links (spdlog::spdlog for SPDLOG_COMPILED_LIB hosts); defaults to
# whatever BasicRHI was configured with.
if(DEFINED BASICRHI_SPDLOG_TARGET)
    set(_org_default_spdlog_target "${BASICRHI_SPDLOG_TARGET}")
else()
    set(_org_default_spdlog_target "spdlog::spdlog_header_only")
endif()
set(OPENRENDERGRAPH_SPDLOG_TARGET "${_org_default_spdlog_target}" CACHE STRING "spdlog target linked by OpenRenderGraph")
find_package(flecs CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS container_hash)
if(OPENRENDERGRAPH_ENABLE_DEBUG_UI)
    find_package(imgui CONFIG REQUIRED)
    find_package(implot CONFIG REQUIRED)
    find_package(slang CONFIG REQUIRED)
endif()

add_library(OpenRenderGraph STATIC
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Contributor/ContributorRegistry.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Contributor/ExtensionRegistry.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Contributor/ContributorGraphHost.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Contributor/ContributorRuntime.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Host/PersistentGraphHost.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/OpenRenderGraph/PersistentGraphHost.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/DescriptorHeap.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/PublicationBindingBundle.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/PreparedTablePublisher.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/LatchBlock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/StagedUploadBatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/PreparedTablePublisher.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/RenderGraph/CompileTelemetry.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/PublicationBindingBundle.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/PreparedInvocationArena.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/External/ContributorRuntime.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/CommandListPool.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/QueueRegistry.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/ImmediateExecution/ImmediateCommandList.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/MemoryMetadataAdapter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/MemoryIntrospectionAPI.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/MemoryIntrospectionBackend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultStatisticsService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultUploadService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/CopyQueueUploadService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultUploadPolicyService.cpp"
    
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultReadbackService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultDescriptorService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/ThreadPoolTaskService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/Runtime/ThreadPoolTaskService.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/DefaultRenderGraphSettingsService.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/Runtime/RuntimeDevice.cpp"
    
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/PassBuilders.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Render/PassExecutionContext.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/RenderGraph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/ExperimentalGraphCompiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/PersistentGraph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/RenderGraphPersistent.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/GraphReplay.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/RenderGraph/GraphReplay.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Render/RenderGraph/PersistentGraph.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/RenderPasses/Base/PersistentTypedPass.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/ExperimentalExecutionState.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/InteropAllocator.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/SharedAliasPoolManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/RenderGraphCompile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/FrameRecording.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/FramePlanning.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/RenderGraphCompileProfile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/Aliasing/RenderGraphAliasingSubsystem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Render/RenderGraph/Aliasing/RenderGraphAliasingAlgorithms.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/CommandRecordingManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/Singletons/DeviceManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/Singletons/DescriptorHeapManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/Singletons/UploadManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/UploadInstance.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/Singletons/ReadbackManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Managers/Singletons/StatisticsManager.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/ExternalBackingResource.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/ExternalBufferResource.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/ExternalTextureResource.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/PixelBuffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/Buffers/Buffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/Buffers/DynamicBufferBase.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/ResourceStateTracker.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/GPUBacking/GPUBufferBacking.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Resources/GPUBacking/GPUTextureBacking.cpp"
)

add_library(OpenRenderGraph::OpenRenderGraph ALIAS OpenRenderGraph)

target_precompile_headers(OpenRenderGraph PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h")

set_target_properties(OpenRenderGraph PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(OpenRenderGraph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"

)

target_link_libraries(OpenRenderGraph
    PUBLIC
        BasicRHI::BasicRHI
        BasicTelemetry::Core
        BasicTelemetry::Tracy
        flecs::flecs
        Boost::container_hash
    PRIVATE
        ${OPENRENDERGRAPH_SPDLOG_TARGET}
)

if(OPENRENDERGRAPH_ENABLE_DEBUG_UI)
    add_library(OpenRenderGraphDebugUI STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/DebugUI/MemoryIntrospectionWidget.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/DebugUI/MemoryViewWidget.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/DebugUI/RenderGraphInspector.cpp")
    add_library(OpenRenderGraph::DebugUI ALIAS OpenRenderGraphDebugUI)
    target_compile_features(OpenRenderGraphDebugUI PUBLIC cxx_std_23)
    target_include_directories(OpenRenderGraphDebugUI PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/include")
    if(MSVC)
        target_compile_options(OpenRenderGraphDebugUI PRIVATE /utf-8)
    endif()
    target_link_libraries(OpenRenderGraphDebugUI PUBLIC OpenRenderGraph::OpenRenderGraph
        imgui::imgui implot::implot slang::slang)
endif()

target_compile_definitions(OpenRenderGraph
    PUBLIC
        BUILD_TYPE_DEBUG=0
        BUILD_TYPE_RELEASE=1
        BUILD_TYPE_RELEASE_DEBUG=2
        $<$<CONFIG:Debug>:BUILD_TYPE=BUILD_TYPE_DEBUG>
        $<$<CONFIG:Release>:BUILD_TYPE=BUILD_TYPE_RELEASE>
        $<$<CONFIG:RelWithDebInfo>:BUILD_TYPE=BUILD_TYPE_RELEASE_DEBUG>
)
target_compile_definitions(OpenRenderGraph PRIVATE NOMINMAX)

if (MSVC)
    target_compile_options(OpenRenderGraph PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
    # RenderGraph.cpp exceeds the default COFF section limit; hosts that embed ORG
    # standalone do not necessarily set /bigobj globally.
    target_compile_options(OpenRenderGraph PRIVATE /bigobj)
endif()

if(OPENRENDERGRAPH_BUILD_TESTS)
    enable_testing()
    add_executable(OpenRenderGraphPersistentTests tests/PersistentGraphTests.cpp tests/GraphReplayRunner.h)
    target_compile_features(OpenRenderGraphPersistentTests PRIVATE cxx_std_23)
    target_compile_definitions(OpenRenderGraphPersistentTests PRIVATE
        ORG_REPLAY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(OpenRenderGraphPersistentTests PRIVATE OpenRenderGraph)
    add_test(NAME OpenRenderGraphPersistentTests COMMAND OpenRenderGraphPersistentTests)
    set_tests_properties(OpenRenderGraphPersistentTests PROPERTIES
        ENVIRONMENT "ORG_GRAPH_REPLAY_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/persistent_test_snapshot.orggraph")
    add_executable(OpenRenderGraphHeadlessBenchmark tests/HeadlessGraphBenchmark.cpp tests/GraphReplayRunner.h)
    target_compile_features(OpenRenderGraphHeadlessBenchmark PRIVATE cxx_std_23)
    target_link_libraries(OpenRenderGraphHeadlessBenchmark PRIVATE OpenRenderGraph)
    add_executable(OpenRenderGraphRecordingRecipeTests tests/RecordingRecipeTests.cpp)
    target_compile_features(OpenRenderGraphRecordingRecipeTests PRIVATE cxx_std_23)
    target_compile_definitions(OpenRenderGraphRecordingRecipeTests PRIVATE NOMINMAX)
    if(MSVC)
        target_compile_options(OpenRenderGraphRecordingRecipeTests PRIVATE /utf-8)
    endif()
    target_link_libraries(OpenRenderGraphRecordingRecipeTests PRIVATE OpenRenderGraph)
    add_test(NAME OpenRenderGraphRecordingRecipeTests COMMAND OpenRenderGraphRecordingRecipeTests)
    # Offscreen suites that create D3D12 (WARP) devices exist only when the D3D12 backend is built.
    set(_org_d3d12_tests OFF)
    if(NOT DEFINED BASICRHI_ENABLE_D3D12 OR BASICRHI_ENABLE_D3D12) # packaged BasicRHI: assume D3D12
        set(_org_d3d12_tests ON)
    endif()
    add_custom_target(OpenRenderGraphAsyncFrameTests DEPENDS
        OpenRenderGraphFrameContextTests OpenRenderGraphAsyncCompileTests
        $<$<BOOL:${_org_d3d12_tests}>:OpenRenderGraphExternalResourceTests>)
    add_executable(OpenRenderGraphFrameContextTests tests/FrameContextTests.cpp)
    target_compile_features(OpenRenderGraphFrameContextTests PRIVATE cxx_std_23)
    target_include_directories(OpenRenderGraphFrameContextTests PRIVATE include)
    target_compile_definitions(OpenRenderGraphFrameContextTests PRIVATE NOMINMAX)
    add_test(NAME OpenRenderGraphFrameContextTests COMMAND OpenRenderGraphFrameContextTests)
    add_executable(OpenRenderGraphAsyncCompileTests tests/AsyncCompileTests.cpp tests/FramePlanningTests.cpp)
    target_compile_features(OpenRenderGraphAsyncCompileTests PRIVATE cxx_std_23)
    target_include_directories(OpenRenderGraphAsyncCompileTests PRIVATE include)
    target_link_libraries(OpenRenderGraphAsyncCompileTests PRIVATE
        OpenRenderGraph BasicTelemetry::Core BasicTelemetry::Tracy
        BasicRHI::BasicRHI ${OPENRENDERGRAPH_SPDLOG_TARGET})
    target_compile_definitions(OpenRenderGraphAsyncCompileTests PRIVATE NOMINMAX)
    add_test(NAME OpenRenderGraphAsyncCompileTests COMMAND OpenRenderGraphAsyncCompileTests)
    if(_org_d3d12_tests)
    add_executable(OpenRenderGraphExternalResourceTests tests/ExternalResourceTests.cpp tests/FrameRecordingTests.cpp tests/FrameRetirementTests.cpp tests/ProgramVersionTests.cpp tests/ReadbackCaptureTests.cpp)
    target_include_directories(OpenRenderGraphExternalResourceTests PRIVATE src/include)
    if(WIN32)
        # BasicRHI is a static library and its D3D12 object references the
        # Streamline C API. Standalone tests do not load the host interposer,
        # so provide the same no-op symbols as the BasicRHI smoke test.
        target_sources(OpenRenderGraphExternalResourceTests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRHI/tests/StreamlineStubs.cpp")
        if(TARGET Streamline::headers)
            target_link_libraries(OpenRenderGraphExternalResourceTests PRIVATE Streamline::headers)
        endif()
    endif()
    if(MSVC)
        target_compile_options(OpenRenderGraphExternalResourceTests PRIVATE /utf-8)
    endif()
    target_link_libraries(OpenRenderGraphExternalResourceTests PRIVATE OpenRenderGraph BasicRHI::BasicRHI d3d12 dxgi d3dcompiler)
    if(WIN32 AND TARGET Pix::headers)
        find_library(OPENRENDERGRAPH_PIX_RUNTIME NAMES WinPixEventRuntime
            HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer" REQUIRED)
        target_link_libraries(OpenRenderGraphExternalResourceTests PRIVATE "${OPENRENDERGRAPH_PIX_RUNTIME}")
    endif()
    add_test(NAME OpenRenderGraphExternalResourceTests COMMAND OpenRenderGraphExternalResourceTests)
    add_executable(OpenRenderGraphCopyQueueUploadTests tests/CopyQueueUploadServiceTests.cpp)
    target_include_directories(OpenRenderGraphCopyQueueUploadTests PRIVATE src/include)
    if(WIN32)
        target_sources(OpenRenderGraphCopyQueueUploadTests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRHI/tests/StreamlineStubs.cpp")
        if(TARGET Streamline::headers)
            target_link_libraries(OpenRenderGraphCopyQueueUploadTests PRIVATE Streamline::headers)
        endif()
    endif()
    if(MSVC)
        target_compile_options(OpenRenderGraphCopyQueueUploadTests PRIVATE /utf-8)
    endif()
    target_link_libraries(OpenRenderGraphCopyQueueUploadTests PRIVATE OpenRenderGraph BasicRHI::BasicRHI d3d12 dxgi d3dcompiler)
    if(WIN32 AND TARGET Pix::headers)
        target_link_libraries(OpenRenderGraphCopyQueueUploadTests PRIVATE "${OPENRENDERGRAPH_PIX_RUNTIME}")
    endif()
    add_test(NAME OpenRenderGraphCopyQueueUploadTests COMMAND OpenRenderGraphCopyQueueUploadTests)
    add_custom_target(OpenRenderGraphOffscreenTests DEPENDS OpenRenderGraphExternalResourceTests)
    if(WIN32)
        # Keep the offscreen suite runnable from its build directory without a
        # deployed renderer or MO2 runtime directory on PATH.
        foreach(runtime_dll
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/WinPixEventRuntime.dll"
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/dxcompiler.dll"
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/dxil.dll"
            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/hlsl.dll")
            if(EXISTS "${runtime_dll}")
                add_custom_command(TARGET OpenRenderGraphExternalResourceTests POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${runtime_dll}"
                        "$<TARGET_FILE_DIR:OpenRenderGraphExternalResourceTests>")
            endif()
        endforeach()
    endif()
    endif() # _org_d3d12_tests
    add_executable(OpenRenderGraphContributorHeaderC tests/ContributorHeaderCompile.c)
    set_target_properties(OpenRenderGraphContributorHeaderC PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    target_include_directories(OpenRenderGraphContributorHeaderC PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME OpenRenderGraphContributorHeaderC COMMAND OpenRenderGraphContributorHeaderC)
    add_executable(OpenRenderGraphContributorRegistryTests tests/ContributorRegistryTests.cpp)
    target_compile_features(OpenRenderGraphContributorRegistryTests PRIVATE cxx_std_23)
    target_link_libraries(OpenRenderGraphContributorRegistryTests PRIVATE OpenRenderGraph)
    add_test(NAME OpenRenderGraphContributorRegistryTests COMMAND OpenRenderGraphContributorRegistryTests)

	add_executable(OpenRenderGraphDeviceRegistryTests tests/DeviceRegistryTests.cpp)
	target_link_libraries(OpenRenderGraphDeviceRegistryTests PRIVATE OpenRenderGraph::OpenRenderGraph)
	add_test(NAME OpenRenderGraphDeviceRegistryTests COMMAND OpenRenderGraphDeviceRegistryTests)
    # Persistent execution on an adopted host VkDevice (DXVK embedding model).
    if(BASICRHI_ENABLE_VULKAN AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/../BasicRHI/cmake/BasicRHIShaderFlags.cmake")
        include("${CMAKE_CURRENT_LIST_DIR}/../BasicRHI/cmake/BasicRHIShaderFlags.cmake")
        if(BASICRHI_DXC_EXECUTABLE)
            set(_org_persistent_write_spv "${CMAKE_CURRENT_BINARY_DIR}/test_shaders/PersistentHostWrite.spv")
            basicrhi_compile_spirv(OUTPUT "${_org_persistent_write_spv}"
                SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tests/shaders/PersistentHostWrite.hlsl"
                ENTRY main PROFILE cs_6_6)
            add_executable(OpenRenderGraphPersistentVulkanHostTests tests/PersistentVulkanHostTests.cpp "${_org_persistent_write_spv}")
            target_compile_features(OpenRenderGraphPersistentVulkanHostTests PRIVATE cxx_std_23)
            target_include_directories(OpenRenderGraphPersistentVulkanHostTests PRIVATE src/include)
            target_compile_definitions(OpenRenderGraphPersistentVulkanHostTests PRIVATE NOMINMAX
                ORG_TEST_PERSISTENT_WRITE_SPV="${_org_persistent_write_spv}")
            if(MSVC)
                target_compile_options(OpenRenderGraphPersistentVulkanHostTests PRIVATE /utf-8)
            endif()
            target_link_libraries(OpenRenderGraphPersistentVulkanHostTests PRIVATE OpenRenderGraph BasicRHI::BasicRHI)
            if(WIN32 AND TARGET Streamline::headers)
                # BasicRHI references the Streamline C API; this test does not load the interposer.
                target_sources(OpenRenderGraphPersistentVulkanHostTests PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRHI/tests/StreamlineStubs.cpp")
                target_link_libraries(OpenRenderGraphPersistentVulkanHostTests PRIVATE Streamline::headers)
            endif()
            if(WIN32 AND TARGET Pix::headers)
                find_library(OPENRENDERGRAPH_PIX_RUNTIME NAMES WinPixEventRuntime
                    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer" REQUIRED)
                target_link_libraries(OpenRenderGraphPersistentVulkanHostTests PRIVATE "${OPENRENDERGRAPH_PIX_RUNTIME}")
                if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/WinPixEventRuntime.dll")
                    add_custom_command(TARGET OpenRenderGraphPersistentVulkanHostTests POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${CMAKE_CURRENT_SOURCE_DIR}/../BasicRenderer/WinPixEventRuntime.dll"
                            "$<TARGET_FILE_DIR:OpenRenderGraphPersistentVulkanHostTests>")
                endif()
            endif()
            add_test(NAME OpenRenderGraphPersistentVulkanHostTests COMMAND OpenRenderGraphPersistentVulkanHostTests)
        endif()
    endif()
    if(WIN32 AND BASICRENDERER_TEST_RUNTIME_DIR)
        set_tests_properties(
            OpenRenderGraphAsyncCompileTests
            OpenRenderGraphContributorHeaderC
            OpenRenderGraphContributorRegistryTests
            OpenRenderGraphDeviceRegistryTests
            PROPERTIES
                WORKING_DIRECTORY "${BASICRENDERER_TEST_RUNTIME_DIR}"
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${BASICRENDERER_TEST_RUNTIME_DIR}")
    endif()
    if(_org_d3d12_tests)
        set_tests_properties(OpenRenderGraphExternalResourceTests PROPERTIES
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:OpenRenderGraphExternalResourceTests>"
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:OpenRenderGraphExternalResourceTests>")
        if(WIN32 AND BASICRENDERER_TEST_RUNTIME_DIR)
            set_tests_properties(OpenRenderGraphExternalResourceTests PROPERTIES
                WORKING_DIRECTORY "${BASICRENDERER_TEST_RUNTIME_DIR}"
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${BASICRENDERER_TEST_RUNTIME_DIR}")
        endif()
    endif()
endif()

if(OPENRENDERGRAPH_ENABLE_D3D11_INTEROP)
    if(NOT WIN32)
        message(FATAL_ERROR "OPENRENDERGRAPH_ENABLE_D3D11_INTEROP is only supported on Windows")
    endif()
    if(NOT TARGET Microsoft::CppWinRT)
        find_package(cppwinrt CONFIG REQUIRED)
    endif()
    add_library(OpenRenderGraphD3D11Interop STATIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/Interop/D3D11Interop.cpp")
    add_library(OpenRenderGraph::D3D11Interop ALIAS OpenRenderGraphD3D11Interop)
    target_compile_features(OpenRenderGraphD3D11Interop PUBLIC cxx_std_23)
    target_compile_definitions(OpenRenderGraphD3D11Interop PRIVATE NOMINMAX)
    target_include_directories(OpenRenderGraphD3D11Interop PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(OpenRenderGraphD3D11Interop PUBLIC BasicRHI::BasicRHI
        PRIVATE Microsoft::CppWinRT d3d11 d3d12 dxgi)
    install(TARGETS OpenRenderGraphD3D11Interop EXPORT OpenRenderGraphTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

install(TARGETS OpenRenderGraph
    EXPORT OpenRenderGraphTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT OpenRenderGraphTargets
    FILE OpenRenderGraphTargets.cmake
    NAMESPACE OpenRenderGraph::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/OpenRenderGraph"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OpenRenderGraphConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenRenderGraphConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenRenderGraphConfig.cmake"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/OpenRenderGraph"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OpenRenderGraphConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenRenderGraphConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/OpenRenderGraph"
)
