import auxiliary.pointcloud_processor as pointcloud_processor


class Augmenter(object):
    def __init__(self, translation=True, rotation_axis=[], rotation_3D=True, anisotropic_scaling=True, flips=[]):
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
