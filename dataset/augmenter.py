import dataset.pointcloud_processor as pointcloud_processor


class Augmenter(object):
    """
    Class defining data augmentation for a batch of points.
    Author : Thibault Groueix 01.11.2019
    """
    def __init__(self, translation=False, rotation_axis=[], rotation_3D=False, anisotropic_scaling=False, flips=[]):
        self.translation = translation
        self.rotation_axis = rotation_axis
        self.rotation_3D = rotation_3D
        self.anisotropic_scaling = anisotropic_scaling
        self.flips = flips

    def __call__(self, points):
        operation = pointcloud_processor.DataAugmentation(points)
        for axis in self.rotation_axis:
            operation.random_axial_rotation(axis=axis)
        if self.rotation_3D:
            operation.random_rotation()
        if self.anisotropic_scaling:
            operation.random_anisotropic_scaling()
        if len(self.flips) > 0:
            operation.random_flips(self.flips)
        if self.translation:
            operation.random_translation()
