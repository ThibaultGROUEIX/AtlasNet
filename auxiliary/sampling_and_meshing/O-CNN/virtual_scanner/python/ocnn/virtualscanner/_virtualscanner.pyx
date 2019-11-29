from ocnn.virtualscanner cimport _virtualscanner_extern
from libcpp.string cimport string
from libcpp cimport bool
from ocnn.virtualscanner.scanner_settings import ScannerSettings

cdef class VirtualScanner:
    """
        Creates points/normals file from obj or off files
    """
    cdef _virtualscanner_extern.VirtualScanner c_scanner

    def __cinit__(self, filepath, int view_num=6, bool flags=False, bool normalize=False):
        """
            Scans obj/off file into a points/normal format

            Args:
                filepath (str): File path of obj/off file to convert.
                view_num (int): The number of views for scanning
                flags (bool): Indicate whether to output normal flipping flag
                normalize (bool): Indicate whether to normalize input mesh
        """

        cdef string stl_string = filepath.encode('UTF-8')
        with nogil:
            self.c_scanner.scanning(stl_string, view_num, flags, normalize)

    @classmethod
    def from_scanner_settings(self, filepath, scanner_settings):
        """
            Scans obj/off file into a points/normal format

            Args:
                filepath (str): File path of obj/off file to convert.
                scanner_settings (ScannerSettings): Virtual scanner settings.
        """
        return VirtualScanner(filepath=filepath,
                              view_num=scanner_settings.view_num,
                              flags=scanner_settings.flags,
                              normalize=scanner_settings.normalize)

    def save(self, output_path):
        """
           Saves out to points/normals file.

           Args:
               output_path (str): Path where to save points file.
        """

        cdef string stl_string = output_path.encode('UTF-8')
        with nogil:
            self.c_scanner.save_binary(stl_string)
