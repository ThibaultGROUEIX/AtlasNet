find_package(VirtualScanner)
if(NOT TARGET OCNN::VirtualScanner)
    message("VirtualScanner library not found")
    add_subdirectory(../../../cpp/libvirtualscanner "${CMAKE_CURRENT_BINARY_DIR}/_libvirtualscanner")
endif()

add_cython_target(_virtualscanner CXX)

add_library(_virtualscanner MODULE ${_virtualscanner})
python_extension_module(_virtualscanner)

if (VCPKG_TOOLCHAIN)
    find_package(CGAL)
    get_target_property(CGAL_DLL CGAL::CGAL IMPORTED_LOCATION_RELEASE)
    install(FILES  ${CGAL_DLL} DESTINATION python/ocnn/virtualscanner)
endif()
target_link_libraries(_virtualscanner OCNN::VirtualScanner)
install(TARGETS _virtualscanner LIBRARY DESTINATION python/ocnn/virtualscanner)
