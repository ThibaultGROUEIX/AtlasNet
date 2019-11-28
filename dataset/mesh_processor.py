import numpy as np
import trimesh
import pymesh


def save_pointcloud_ply(points, path):
    """
    Invoked with a torch tensor
    :param points:
    :param path:
    :return:
    """
    mesh = pymesh.form_mesh(vertices=points.cpu().numpy(), faces=np.array([[0, 1, 2]]))
    pymesh.save_mesh(path, mesh, ascii=True)


def test_orientation(input_mesh):
    """
    This fonction tests wether widest axis of the input mesh is the Z axis
    input mesh
    output : boolean or warning
    """
    point_set = input_mesh.vertices
    bbox = np.array([[np.max(point_set[:, 0]), np.max(point_set[:, 1]), np.max(point_set[:, 2])],
                     [np.min(point_set[:, 0]), np.min(point_set[:, 1]), np.min(point_set[:, 2])]])
    extent = bbox[0] - bbox[1]
    if not np.argmax(np.abs(extent)) == 1:
        print(
            "The widest axis is not the Y axis, you should make sure the mesh is aligned on the Y axis for the autoencoder to work (check out the example in /data)")
    return


def clean(input_mesh, prop=None):
    """
    This function remove faces, and vertex that doesn't belong to any face. Intended to be used before a feed forward pass in pointNet
    Input : mesh
    output : cleaned mesh
    """
    print("cleaning ...")
    print("number of point before : ", np.shape(input_mesh.vertices)[0])
    pts = input_mesh.vertices
    faces = input_mesh.faces
    faces = faces.reshape(-1)
    unique_points_index = np.unique(faces)
    unique_points = pts[unique_points_index]

    print("number of point after : ", np.shape(unique_points)[0])
    mesh = trimesh.Trimesh(vertices=unique_points, faces=np.array([[0, 0, 0]]), process=False)
    if prop is not None:
        new_prop = prop[unique_points_index]
        return mesh, new_prop
    else:
        return mesh


def center(input_mesh):
    """
    This function center the input mesh using it's bounding box
    Input : mesh
    output : centered mesh and translation vector
    """
    bbox = np.array(
        [[np.max(input_mesh.vertices[:, 0]), np.max(input_mesh.vertices[:, 1]), np.max(input_mesh.vertices[:, 2])],
         [np.min(input_mesh.vertices[:, 0]), np.min(input_mesh.vertices[:, 1]), np.min(input_mesh.vertices[:, 2])]])

    tranlation = (bbox[0] + bbox[1]) / 2
    points = input_mesh.vertices - tranlation
    mesh = trimesh.Trimesh(vertices=points, faces=input_mesh.faces, process=False)
    return mesh, tranlation


def scale(input_mesh, mesh_ref):
    """
    This function scales the input mesh to have the same volume as a reference mesh Intended to be used before a feed forward pass in pointNet
    Input : file path
    mesh_ref : reference mesh path
    output : scaled mesh
    """
    area = np.power(mesh_ref.volume / input_mesh.volume, 1.0 / 3)
    mesh = trimesh.Trimesh(vertices=input_mesh.vertices * area, faces=input_mesh.faces, process=False)
    return mesh, area


def fix_height(input_mesh, mesh_ref):
    """
    This function scales the input mesh to have the same volume as a reference mesh Intended to be used before a feed forward pass in pointNet
    Input : file path
    mesh_ref : reference mesh path
    output : scaled mesh
    """
    div_ref = np.max(mesh_ref.vertices[:, 1]) - np.min(mesh_ref.vertices[:, 1])
    div = np.max(input_mesh.vertices[:, 1]) - np.min(input_mesh.vertices[:, 1])
    ratio = div_ref / div
    mesh = trimesh.Trimesh(vertices=input_mesh.vertices * ratio, faces=input_mesh.faces, process=False)
    return mesh, ratio


def uniformize(input):
    input = pymesh.form_mesh(input.vertices, input.faces)
    input, _ = pymesh.split_long_edges(input, 0.005)
    return input


def rot(input_mesh, theta=np.pi / 2):
    # rotation around X axis of angle theta
    point = input_mesh.vertices
    rot_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    point_set = point.dot(np.transpose(rot_matrix, (1, 0)))
    # center the rotated mesh
    bbox = np.array([[np.max(point_set[:, 0]), np.max(point_set[:, 1]), np.max(point_set[:, 2])],
                     [np.min(point_set[:, 0]), np.min(point_set[:, 1]), np.min(point_set[:, 2])]])

    tranlation = (bbox[0] + bbox[1]) / 2
    point_set = point_set - tranlation

    mesh = trimesh.Trimesh(vertices=point_set, faces=input_mesh.faces, process=False)
    return mesh


def get_vertex_normalised_area(mesh):
    # input : pymesh mesh
    # output : Numpy array #vertex summing to 1
    num_vertices = mesh.vertices.shape[0]
    print("num_vertices", num_vertices)
    a = mesh.vertices[mesh.faces[:, 0]]
    b = mesh.vertices[mesh.faces[:, 1]]
    c = mesh.vertices[mesh.faces[:, 2]]
    cross = np.cross(a - b, a - c)
    area = np.sqrt(np.sum(cross ** 2, axis=1))
    prop = np.zeros((num_vertices))
    prop[mesh.faces[:, 0]] = prop[mesh.faces[:, 0]] + area
    prop[mesh.faces[:, 1]] = prop[mesh.faces[:, 1]] + area
    prop[mesh.faces[:, 2]] = prop[mesh.faces[:, 2]] + area
    return prop / np.sum(prop)
