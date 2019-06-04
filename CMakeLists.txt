cmake_minimum_required(VERSION 3.8)

project(MAGSAC LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 3.3 REQUIRED)

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
include_directories (
	${PROJECT_SOURCE_DIR}/include
)

# ==============================================================================
# Structure: Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_MAGSAC
	"include/*.h"
	"include/*.cpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MAGSAC 
	"src/*.cpp"
)

add_executable(${PROJECT_NAME}
	${HDRS_MAGSAC}
	${SRCS_MAGSAC}
)

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
)

# ==============================================================================
# Structure: Applications
# ==============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

