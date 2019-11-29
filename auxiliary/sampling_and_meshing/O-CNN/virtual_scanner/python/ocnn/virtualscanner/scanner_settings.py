""" Module to manage ScannerSettings """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml


class ScannerSettings:
    """ Loads, writes and compares ScannerSettings """

    def __init__(self,
                 view_num=6,
                 flags=False,
                 normalize=False):
        """ Initializes ScannerSettings
        Args:
          view_num (int): The number of view points to scan from.
          flags (bool): Indicate whether to ouput normal flipping flag.
          normalize (bool): Normalize maximum extents of mesh to 1.
        """
        self.view_num = view_num
        self.flags = flags
        self.normalize = normalize

    @classmethod
    def from_yaml(cls, yml_filepath):
        """ Creates ScannerSettings from YAML
        Args:
          yml_filepath: Path to yml config file.
        """
        with open(yml_filepath, 'r') as yml_file:
            config = yaml.load(yml_file)

        parameters = config['scanner_settings']
        return cls(view_num=parameters['view_num'],
                   flags=parameters['flags'],
                   normalize=parameters['normalize'])

    def write_yaml(self, yml_filepath):
        """ Writes ScannerSettings to YAML
        Args:
          yml_filepath: Filepath to output settings
        """
        data = {'scanner_settings': {
            'view_num': self.view_num,
            'flags': self.flags,
            'normalize': self.normalize}}
        with open(yml_filepath, 'w') as yml_file:
            yaml.dump(data, yml_file, default_flow_style=False)

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return (self.view_num == other.view_num and
                    self.flags == other.flags and
                    self.normalize == other.normalize)
        return NotImplemented

    def __ne__(self, other):
        return not self == other
