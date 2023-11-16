##############################
PROJECT( LicPlateReader )
##############################
cmake_minimum_required (VERSION 3.0)

# Packages
find_package( OpenCV REQUIRED )

# Includes
include_directories(	
	${OPENCV_INCLUDE_DIR}
	${PROJECT_BINARY_DIR}
)

include_directories ("usr/local/include")

link_directories ("/usr/local/lib")

set(SOURCE_FILES 
	src/main.cpp
)


# Add executable and target link libraries
ADD_EXECUTABLE( LicPlateReader ${SOURCE_FILES})
TARGET_LINK_LIBRARIES( LicPlateReader ${OpenCV_LIBS} -lpthread -lX11)


