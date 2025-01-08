cmake_minimum_required(VERSION 3.13)

project(piper C CXX)

file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" piper_version)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # Force compiler to use UTF-8 for IPA constants
  add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
elseif(NOT APPLE)
  # Linux flags
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wl,-rpath,'$ORIGIN'")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

add_executable(piper src/cpp/main.cpp src/cpp/piper.cpp)
add_executable(piper_server src/cpp/server.cpp src/cpp/piper.cpp)
add_executable(test_piper src/cpp/test.cpp src/cpp/piper.cpp)

# NOTE: external project prefix are shortened because of path length restrictions on Windows
# NOTE: onnxruntime is pulled from piper-phonemize

# ---- fmt ---

if(NOT DEFINED FMT_DIR)
  set(FMT_VERSION "10.0.0")
  set(FMT_DIR "${CMAKE_CURRENT_BINARY_DIR}/fi")

  include(ExternalProject)
  ExternalProject_Add(
    fmt_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/f"
    URL "https://github.com/fmtlib/fmt/archive/refs/tags/${FMT_VERSION}.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FMT_DIR}
    CMAKE_ARGS -DFMT_TEST:BOOL=OFF  # Don't build all the tests
  )
  add_dependencies(piper fmt_external)
  add_dependencies(piper_server fmt_external)
  add_dependencies(test_piper fmt_external)
endif()

# ---- spdlog ---

if(NOT DEFINED SPDLOG_DIR)
  set(SPDLOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/si")
  set(SPDLOG_VERSION "1.12.0")
  ExternalProject_Add(
    spdlog_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/s"
    URL "https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${SPDLOG_DIR}
  )
  add_dependencies(piper spdlog_external)
  add_dependencies(piper_server spdlog_external)
  add_dependencies(test_piper spdlog_external)
endif()

# ---- piper-phonemize ---

if(NOT DEFINED PIPER_PHONEMIZE_DIR)
  set(PIPER_PHONEMIZE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pi")
  ExternalProject_Add(
    piper_phonemize_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/p"
    URL "https://github.com/rhasspy/piper-phonemize/archive/refs/heads/master.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PIPER_PHONEMIZE_DIR}
  )
  add_dependencies(piper piper_phonemize_external)
  add_dependencies(piper_server piper_phonemize_external)
  add_dependencies(test_piper piper_phonemize_external)
endif()

if(NOT DEFINED SOUNDTOUCH_DIR)
  set(SOUNDTOUCH_DIR "${CMAKE_CURRENT_BINARY_DIR}/so")
  ExternalProject_Add(
    soundtouch_external
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/so"
    URL "https://gitlab.com/soundtouch/soundtouch/-/archive/2.3.1/soundtouch-2.3.1.zip"
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${SOUNDTOUCH_DIR}
  )
  add_dependencies(piper soundtouch_external)
  add_dependencies(piper_server soundtouch_external)
  add_dependencies(test_piper soundtouch_external)
endif()

# ---- libeigen3-dev ----
if(NOT DEFINED LIBEIGEN_DIR)
if(WIN32)
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_SOURCE_DIR}/eigen3")
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
  set(LIBEIGEN_DIR "${CMAKE_SOURCE_DIR}/eigen3")
else()
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/include/eigen3/include")
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    set(LIBEIGEN_DIR "/usr/include/eigen3")
  endif()
  include_directories(${LIBEIGEN_DIR})
endif()

# # ---- SoundTouch ----
# if(WIN32)
#   set(SOUNDTOUCH_DIR "${CMAKE_SOURCE_DIR}/soundtouch")
#   include_directories(${SOUNDTOUCH_DIR}/include)
#   link_directories(${SOUNDTOUCH_DIR}/lib)
#   find_package(SoundTouch REQUIRED)
#   find_package(SoundTouch)
#   if(SOUNDTOUCH_FOUND)
#     message(STATUS "SoundTouch found!")
#   else()
#     message(STATUS "SoundTouch not found!")
#   endif()
# else()
#   find_package(SoundTouch REQUIRED)
#   set(SOUNDTOUCH_LIB SoundTouch)
# endif()

# message(STATUS "SoundTouch include directory: ${SOUNDTOUCH_DIR}/include")
# message(STATUS "SoundTouch library directory: ${SOUNDTOUCH_DIR}/lib")



target_link_libraries(${PROJECT_NAME} ${SOUNDTOUCH_LIB})
# target_link_libraries(${PROJECT_NAME} ${SOUNDTOUCH_INCLUDE_DIR})

# ---- Declare executable ----

if((NOT MSVC) AND (NOT APPLE))
  # Linux flags
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wl,-rpath,'$ORIGIN'")
  string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
  target_link_libraries(piper -static-libgcc -static-libstdc++)
  target_link_libraries(piper_server -static-libgcc -static-libstdc++)

  set(PIPER_EXTRA_LIBRARIES "pthread")
endif()

target_link_libraries(piper
  fmt
  spdlog
  espeak-ng
  piper_phonemize
  onnxruntime
  SoundTouch
  Eigen3::Eigen
  ${PIPER_EXTRA_LIBRARIES}
)

target_link_directories(piper PUBLIC
  ${FMT_DIR}/lib
  ${SPDLOG_DIR}/lib
  ${SOUNDTOUCH_DIR}/lib
  ${PIPER_PHONEMIZE_DIR}/lib
)

target_include_directories(piper PUBLIC
  ${FMT_DIR}/include
  ${SPDLOG_DIR}/include
  ${PIPER_PHONEMIZE_DIR}/include
  ${LIBEIGEN_DIR}
  ${SOUNDTOUCH_DIR}/include
)

target_compile_definitions(piper PUBLIC _PIPER_VERSION=${piper_version})

# Declare server executable
target_link_libraries(piper_server
  fmt
  spdlog
  espeak-ng
  piper_phonemize
  onnxruntime
  SoundTouch
  Eigen3::Eigen
  ${PIPER_EXTRA_LIBRARIES}
)

target_link_directories(piper_server PUBLIC
  ${FMT_DIR}/lib
  ${SPDLOG_DIR}/lib
  ${PIPER_PHONEMIZE_DIR}/lib
  ${LIBEIGEN_DIR}
  ${SOUNDTOUCH_DIR}/lib
)

target_include_directories(piper_server PUBLIC
  ${FMT_DIR}/include
  ${SPDLOG_DIR}/include
  ${PIPER_PHONEMIZE_DIR}/include
  ${LIBEIGEN_DIR}
  ${SOUNDTOUCH_DIR}/include
)

target_compile_definitions(piper_server PUBLIC _PIPER_VERSION=${piper_version})

# ---- Declare test ----
include(CTest)
enable_testing()
add_test(
  NAME test_piper
  COMMAND test_piper "${CMAKE_SOURCE_DIR}/etc/test_voice.onnx" "${PIPER_PHONEMIZE_DIR}/share/espeak-ng-data" "${CMAKE_CURRENT_BINARY_DIR}/test.wav"
)

target_compile_features(test_piper PUBLIC cxx_std_17)

target_include_directories(
  test_piper PUBLIC
  ${FMT_DIR}/include
  ${SPDLOG_DIR}/include
  ${PIPER_PHONEMIZE_DIR}/include
  ${LIBEIGEN_DIR}
  ${SOUNDTOUCH_DIR}/include
)

target_link_directories(
  test_piper PUBLIC
  ${FMT_DIR}/lib
  ${SPDLOG_DIR}/lib
  ${PIPER_PHONEMIZE_DIR}/lib
  ${LIBEIGEN_DIR}
  ${SOUNDTOUCH_DIR}/lib
)

target_link_libraries(test_piper PUBLIC
  fmt
  spdlog
  espeak-ng
  piper_phonemize
  onnxruntime
  SoundTouch
  Eigen3::Eigen
)

# ---- Declare install targets ----

install(
  TARGETS piper
  DESTINATION ${CMAKE_INSTALL_PREFIX})

# Dependencies
install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/bin/
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  USE_SOURCE_PERMISSIONS  # keep +x
  FILES_MATCHING
  PATTERN "piper_phonemize"
  PATTERN "espeak-ng"
  PATTERN "*.dll"
)

install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/lib/
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING
  PATTERN "*.dll"
  PATTERN "*.so*"
)

install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/share/espeak-ng-data
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  FILES ${PIPER_PHONEMIZE_DIR}/share/libtashkeel_model.ort
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

# ---- Declare install targets for server----

install(
  TARGETS piper_server
  DESTINATION ${CMAKE_INSTALL_PREFIX})

# Dependencies
install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/bin/
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  USE_SOURCE_PERMISSIONS  # keep +x
  FILES_MATCHING
  PATTERN "piper_phonemize"
  PATTERN "espeak-ng"
  PATTERN "*.dll"
)

install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/lib/
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING
  PATTERN "*.dll"
  PATTERN "*.so*"
)

install(
  DIRECTORY ${PIPER_PHONEMIZE_DIR}/share/espeak-ng-data
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

install(
  FILES ${PIPER_PHONEMIZE_DIR}/share/libtashkeel_model.ort
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)