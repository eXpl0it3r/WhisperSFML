cmake_minimum_required(VERSION 3.21)
project(WhisperSFML LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(SFML GIT_REPOSITORY https://github.com/SFML/SFML.git)
FetchContent_MakeAvailable(SFML)

add_executable(WhisperSFML src/main.cpp)
target_link_libraries(WhisperSFML PRIVATE sfml-graphics sfml-audio)
target_compile_features(WhisperSFML PRIVATE cxx_std_17)

if (WIN32 AND BUILD_SHARED_LIBS)
    add_custom_command(TARGET WhisperSFML POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:WhisperSFML> $<TARGET_FILE_DIR:WhisperSFML> COMMAND_EXPAND_LISTS)
endif()

install(TARGETS WhisperSFML)