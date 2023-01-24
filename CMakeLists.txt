cmake_minimum_required(VERSION 3.21)
project(WhisperSFML LANGUAGES CXX)

include(FetchContent)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

FetchContent_Declare(WHISPERCPP GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git)
FetchContent_MakeAvailable(WHISPERCPP)

FetchContent_Declare(SFML GIT_REPOSITORY https://github.com/SFML/SFML.git)
FetchContent_MakeAvailable(SFML)

add_executable(WhisperSFML src/main.cpp)

target_link_libraries(WhisperSFML PRIVATE SFML::Graphics SFML::Audio whisper Threads::Threads)
target_compile_features(WhisperSFML PRIVATE cxx_std_17)

if (WIN32 AND ${BUILD_SHARED_LIBS})
    add_custom_command(TARGET WhisperSFML POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:WhisperSFML> $<TARGET_FILE_DIR:WhisperSFML> COMMAND_EXPAND_LISTS)
endif()

install(TARGETS WhisperSFML)
install(DIRECTORY res DESTINATION ${CMAKE_INSTALL_BINDIR})