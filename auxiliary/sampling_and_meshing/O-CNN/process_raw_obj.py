import randomizePointCloud
import pcSamplingInfRayShapeNet
import sample30kpoints
from easydict import EasyDict
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapenet_path', help="Input Folder", required=True)
    parser.add_argument('--virtualscan', type=str, help='Path to the virtual scanner executable',
                        default='virtual_scanner/build/virtualscanner')
    parser.add_argument('--limit', help="Max number of points in the output cloud", default=0, type=int)

    args = parser.parse_args()
    print("Infinite ray sampling...")
    pcSamplingInfRayShapeNet.shoot_rays(EasyDict(args.__dict__))
    print("Done! Random Shuffling...")
    randomizePointCloud.shuffle_folder(EasyDict(args.__dict__))
    print("Done! Random Sampling...")
    sample30kpoints.test_folder(EasyDict(args.__dict__))
    print("Done!")


if __name__ == "__main__":
    main()
