import visdom
import os
import sys
import time

"""
    Author : Thibault Groueix 01.11.2019
"""


def is_port_in_use(port):
    """
    test if a port is being used or is free to use.
    :param port:
    :return:
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


class Visualizer(object):
    def __init__(self, visdom_port, env, http_port):
        super(Visualizer, self).__init__()
        # Create Visdom Server
        try:
            if not is_port_in_use(visdom_port):
                print(f"Launching new visdom instance in port {visdom_port}")
                cmd = f"{sys.executable} -m visdom.server -p {visdom_port} > /dev/null 2>&1"
                CMD = f'TMUX=0 tmux new-session -d -s visdom_server \; send-keys "{cmd}" Enter'
                print(CMD)
                os.system(CMD)
                time.sleep(2)
        except:
            print("coudn't set up visdom server.")

        try:
            # Create Http Server
            if not is_port_in_use(http_port):
                print(f"Launching new HTTP instance in port {http_port}")
                cmd = f"{sys.executable} -m http.server -p {http_port} > /dev/null 2>&1"
                CMD = f'TMUX=0 tmux new-session -d -s http_server \; send-keys "{cmd}" Enter'
                print(CMD)
                os.system(CMD)
        except:
            print("couldn't set up http server.")

        self.visdom_port = visdom_port
        self.http_port = http_port

        vis = visdom.Visdom(port=visdom_port, env=env)
        self.vis = vis

    def show_pointcloud(self, points, title=None, Y=None):
        """
        :param points: pytorch tensor pointcloud
        :param title:
        :param Y:
        :return:
        """
        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        opts = dict(
            title=title,
            markersize=2,
            xtickmin=-0.7,
            xtickmax=0.7,
            xtickstep=0.3,
            ytickmin=-0.7,
            ytickmax=0.7,
            ytickstep=0.3,
            ztickmin=-0.7,
            ztickmax=0.7,
            ztickstep=0.3)

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=opts)
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=opts
            )

    def show_pointclouds(self, points, title=None, Y=None):
        points = points.squeeze()
        assert points.dim() == 3
        for i in range(points.size(0)):
            self.show_pointcloud(points[i], title=title)

    def show_image(self, img, title=None):
        img = img.squeeze()
        self.vis.image(img, win=title, opts=dict(title=title))
