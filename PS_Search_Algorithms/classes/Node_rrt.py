import math
from mayavi import mlab
import numpy as np
from copy import copy
from Setup.Maze import Maze


class Node:
    """
    RRT Node
    """

    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta  # this takes values between -np.pi and np.pi
        self.path_x = []
        self.path_y = []
        self.path_theta = []  # this takes values between -np.pi and np.pi
        self.parent = None

    def connected(self, conf_space):
        ind = conf_space.coords_to_indices(self.x, self.y, self.theta)
        nn = []
        for ix in [ind[0]-1, ind[0], ind[0]+1]:
            for iy in [ind[1]-1, ind[1], ind[1]+1]:
                for itheta in [ind[2]-1, ind[2], ind[2]+1]:
                    if not conf_space.space[ix, iy, itheta]:
                        nn.append((ix, iy, itheta))
        nn.remove(ind)
        return nn

    def distance(self, node, average_radius):
        return np.sqrt((node.x - self.x) ** 2 +
                       (node.y - self.y) ** 2 +
                       (((node.theta - self.theta + np.pi) % (2 * np.pi) - np.pi) * average_radius) ** 2
                       )

    def calc_distance_and_angles(self, to_node, average_radius):
        dx = to_node.x - self.x
        dy = to_node.y - self.y
        # dtheta = ((to_node.theta - self.theta) % (2 * np.pi) * av_radius)
        dtheta = (((to_node.theta - self.theta + np.pi) % (2 * np.pi) - np.pi) * average_radius)

        r = np.linalg.norm([dx, dy, dtheta])
        azimut = math.atan2(dy, dx)  # 0 to 2pi
        polar = math.acos(dtheta / r)  # 0 to pi

        return r, azimut, polar

    def get_nearest_node(self, node_list, average_radius):
        dlist = [self.distance(node, average_radius) for node in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    def draw_node(self, fig=None, scale_factor=0.2, color=(0, 0, 0)):
        # plot the random point
        # if point is not None:
        mlab.points3d(self.x, self.y, self.theta * average_radius,
                      figure=fig,
                      scale_factor=scale_factor,
                      color=color,
                      )

    def draw_line(self, node, fig=None, line_width=0.2, color=(0, 0, 0)):

        if abs(node.theta - self.theta) > np.pi:
            if node.theta > self.theta:
                upper_node, lower_node = copy(node), copy(self)
            else:
                upper_node, lower_node = copy(self), copy(node)

            d = lower_node.theta % (2 * np.pi) - upper_node.theta
            A = lower_node.theta % (2 * np.pi) - np.pi
            half_way_x = (node.x - self.x) * A / d + self.x
            half_way_y = (node.y - self.y) * A / d + self.y

            # upper_between_node = Node(*[half_way_x, half_way_y, np.pi])
            # lower_between_node = Node(*[half_way_x, half_way_y, -np.pi])
            upper_between_node = Node(*[half_way_x, half_way_y, 2 * np.pi])
            lower_between_node = Node(*[half_way_x, half_way_y, 0])

            mlab.plot3d([upper_node.x, upper_between_node.x],
                        [upper_node.y, upper_between_node.y],
                        [upper_node.theta * average_radius, upper_between_node.theta * average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )
            mlab.plot3d([lower_between_node.x, lower_node.x],
                        [lower_between_node.y, lower_node.y],
                        [lower_between_node.theta * average_radius, lower_node.theta * average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )

        else:
            mlab.plot3d([self.x, node.x], [self.y, node.y], [self.theta * average_radius, node.theta * average_radius],
                        figure=fig,
                        line_width=line_width,
                        color=color,
                        )

        # plot the nodes
        # for node in self.node_list:
        #     if node.parent:
        #         mlab.points3d(node.path_x, node.path_y, node.path_theta,
        #                       figure=fig,
        #                       scale_factor=0.2,
        #                       )
