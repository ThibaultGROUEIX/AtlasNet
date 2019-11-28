import scipy.misc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D  # pycharm auto import
import auxiliary.pointcloud_processor as pointcloud_processor
import torch


def save_rendering_of_points_3D(points, path, dpi=8, color=None):
    if color is None:
        color = 'salmon'
    elif type(color) == torch.Tensor:
        color = color.cpu().squeeze().int().numpy().tolist()

    points = pointcloud_processor.Normalization.normalize_unitL2ball_functional(points).squeeze().cpu().numpy()
    # Normalize points
    p1 = points[:, 0]
    p2 = points[:, 1]
    p3 = points[:, 2]
    # fig = plt.figure(figsize=(20, 20), dpi=6)
    fig = plt.figure(figsize=(20, 20), dpi=dpi)
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    # ax = fig.add_subplot(111, projection='3d',  facecolor='#202124')
    ax.view_init(-30, 30)
    ax.set_xlim3d(-0.52, 0.52)
    ax.set_ylim3d(-0.52, 0.52)
    ax.set_zlim3d(-0.52, 0.52)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.scatter(p3, p1, p2, c=color, alpha=1, s=500)
    plt.grid(b=None)
    plt.axis('off')
    fig.savefig(path, bbox_inches='tight',
                pad_inches=0)
    plt.close('all')


def save_rendering_of_points_2D(points, path, dpi=8):
    points = pointcloud_processor.Normalization.normalize_unitL2ball_functional(points)
    points = points.cpu().squeeze().numpy()
    points[:] = np.round(((points[:] + 1.0) / 2.0) * 27).astype('int')
    render = np.zeros((28, 28))
    render[points[:, 0].astype('int'), points[:, 1].astype('int')] = 1
    scipy.misc.imsave(path, render)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def save_rendering_of_image(points, path):
    # scipy.misc.imsave(path + '.png', np.squeeze(convert_image_np(points)))
    points = points.squeeze()
    if points.size(0) == 3:
        points = points.transpose(0, 1).transpose(1, 2).contiguous()
    scipy.misc.imsave(path, np.squeeze(points.cpu().numpy()))


def save_rendering_of_points(points, path, dpi=10, color=None):
    if points.size(1) == 3:
        save_rendering_of_points_3D(points, path, dpi=dpi, color=color)
    elif points.size(1) == 2:
        save_rendering_of_points_2D(points, path, dpi=dpi)
    else:
        save_rendering_of_image(points, path)
