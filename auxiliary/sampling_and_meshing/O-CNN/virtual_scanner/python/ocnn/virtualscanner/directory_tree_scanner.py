""" Module to output virtual scan a whole directory. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue

import os

from ocnn.virtualscanner.scanner_settings import ScannerSettings
from ocnn.virtualscanner._virtualscanner import VirtualScanner
from threading import Thread


class DirectoryTreeScanner:
    """ Walks a directory and converts off/obj files to points files. """

    def __init__(self, view_num=6, flags=False, normalize=False):
        """ Initializes DirectoryTreeScanner
        Args:
          view_num (int): The number of view points to scan from.
          flags (bool): Indicate whether to ouput normal flipping flag.
           normalize (bool): Normalize maximum extents of mesh to 1.
        """
        self.scanner_settings = ScannerSettings(view_num=view_num,
                                                flags=flags,
                                                normalize=normalize)
        self.scan_queue = Queue()

    def _scan(self):
        """ Creates VirtualScanner object and creates points file from obj/off """
        while True:
            input_path, output_path = self.scan_queue.get()

            print('Scanning {0}'.format(input_path))
            scanner = VirtualScanner.from_scanner_settings(
                input_path,
                self.scanner_settings)
            scanner.save(output_path)
            self.scan_queue.task_done()

    @classmethod
    def from_scanner_settings(cls, scanner_settings):
        """ Create DirectoryTreeScanner from ScannerSettings object
        Args:
          scanner_settings (ScannerSettings): ScannerSettings object
        """
        return cls(view_num=scanner_settings.view_num,
                   flags=scanner_settings.flags,
                   normalize=scanner_settings.normalize)

    def scan_tree(self,
                  input_base_folder,
                  output_base_folder,
                  num_threads=1,
                  output_yaml_filename=''):
        """ Walks directory looking for obj/off files. Outputs points files for
            found obj/off files.

        Args:
          input_base_folder (str): Base folder to scan
          output_base_folder (str): Base folder to output points files in
            mirrored directory structure.
          num_threads (int): Number of threads to use to convert obj/off
            to points
          output_yaml_filename (str): If specified, saves scanner
            settings to given filename in base folder.
        """

        if not os.path.exists(output_base_folder):
            os.mkdir(output_base_folder)
        elif os.listdir(output_base_folder):
            raise RuntimeError('Ouput folder {0} must be empty'.format(
                output_base_folder))

        if output_yaml_filename:
            self.scanner_settings.write_yaml(
                os.path.join(output_base_folder, output_yaml_filename))

        for _ in range(num_threads):
            scan_thread = Thread(target=self._scan)
            scan_thread.daemon = True
            scan_thread.start()

        for root, _, files in os.walk(input_base_folder):
            rel_path = os.path.relpath(root, input_base_folder)

            output_folder = os.path.join(output_base_folder, rel_path)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            for filename in files:
                basename, extension = os.path.splitext(filename)
                extension = extension.lower()
                if extension == '.obj' or extension == '.off':
                    outfilename = basename + '.points'
                    input_path = os.path.join(root, filename)
                    output_path = os.path.join(output_folder, outfilename)
                    self.scan_queue.put((input_path, output_path))
        self.scan_queue.join()
