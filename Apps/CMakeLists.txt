cmake_minimum_required(VERSION 3.0)
project(BioFW.Thermo.Eval)

file(GLOB_RECURSE BioFW.Themo.Eval.Sources "eval/*.cpp")

add_executable(${PROJECT_NAME} ${BioFW.Themo.Eval.Sources})

target_include_directories(${PROJECT_NAME} 
	PUBLIC include
)

find_package( OpenCV REQUIRED )



target_link_libraries(${PROJECT_NAME}
	PocoFoundation
	PocoUtil
	PocoData
	BioFW.Core
	BioFW.OpenCV
	BioFW.Standard
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)


project(BioFW.Thermo)

file(GLOB_RECURSE BioFW.Themo.Sources "thermo/*.cpp")

add_executable(${PROJECT_NAME} ${BioFW.Themo.Sources})

target_include_directories(${PROJECT_NAME} 
	PUBLIC include
)

find_package( OpenCV REQUIRED )


target_link_libraries(${PROJECT_NAME}
	PocoFoundation
	PocoUtil
	PocoData
	BioFW.Core
	BioFW.OpenCV
	BioFW.Standard
	${OpenCV_LIBS}
)

INSTALL(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)