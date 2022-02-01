"""

Path planning Sample Code with Randomized Rapidly-Exploring Random
Trees with sobol low discrepancy sampler(RRTSobol).

The goal of low discrepancy samplers is to generate a sequence of points that
optimizes a criterion called dispersion.  Intuitively, the idea is to place
samples to cover the exploration space in a way that makes the largest
uncovered area be as small as possible.  This generalizes of the idea of grid
resolution.  For a grid, the resolution may be selected by defining the step
size for each axis.  As the step size is decreased, the resolution increases.
If a grid-based motion planning algorithm can increase the resolution
arbitrarily, it becomes resolution complete.  Dispersion can be considered as a
powerful generalization of the notion of resolution.

Taken from
LaValle, Steven M. Planning algorithms. Cambridge university press, 2006.

authors:
    First implementation AtsushiSakai(@Atsushi_twi)
Rojas (rafaelrojasmiliani@gmail.com)


"""
from progressbar import progressbar
import math
import random
from PhaseSpaces import PhaseSpace
from Setup.Maze import start, end, Maze
from PS_Search_Algorithms.classes.Node_rrt import Node

from mayavi import mlab
import os
import numpy as np
from Analysis.GeneralFunctions import graph_dir

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    def __init__(self,
                 start,
                 end,
                 conf_space,
                 sampling_area,
                 expand_dis=1,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=100000,
                 average_radius=average_radius,
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        end_screen:Goal Position [x,y]
        conf_space:Configuration Space [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.sampling_area = sampling_area
        self.expand_dis = expand_dis  # how far you walk towards the random point chosen (stepsize), until you chose another point
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate  # percent of time we place our sample point on our goal
        self.max_iter = max_iter
        self.conf_space = conf_space  # 1, if there is a collision, otherwise 0
        self.node_list = []
        self.average_radius = average_radius

        # check whether start and end_screen are in conf_space
        self.start = Node(*start)
        if self.collision(self.start.x, self.start.y, self.start.theta):
            print('Your start is not in conf_space!')
        self.end = Node(*end)
        if self.collision(self.end.x, self.end.y, self.end.theta):
            print('Your end is not in conf_space!')

    def planning(self, animation=True, fig=None):
        """
        rrt path planning
        Create node list, with only the start,
        iteratively get a random_node,
        find the node nearest to the random_node node in the node_list,
        steer from the nearest node towards the random_node and stop after walking distance expand_dis,
        this is the position of your new_node.
        Check for collisions on the path to the new node.
        If there is a collision, disregard the new_node
        If there is no collision, add the new_node to the node_list, draw it, check whether it
        is close enough to the final node.
        If no, continue iterating.
        If yes, check if the path to the new node is without collisions
        """

        self.node_list = [self.start]
        for i in progressbar(range(self.max_iter)):
            rnd_node = self.get_random_node()
            nearest_node = rnd_node.get_nearest_node(self.node_list, self.average_radius)

            new_node = self.steer(nearest_node, rnd_node, extend_length=self.expand_dis)

            if not self.collision_on_path(new_node):
                self.node_list.append(new_node)

                if animation:
                    new_node.draw_node(fig=fig, average_radius=self.average_radius, color=(1, 0, 0))
                    new_node.draw_line(new_node.parent, fig=fig, average_radius=self.average_radius, color=(0, 0, 0))

                if self.end.x - self.node_list[-1].x <= self.expand_dis:
                    final_node = self.steer(self.node_list[-1], self.end, extend_length=self.expand_dis)
                    if not self.collision_on_path(final_node):
                        self.end.y = self.node_list[-1].y
                        self.end.theta = self.node_list[-1].theta
                        self.end.draw_node(fig=fig, scale_factor=0.5, color=(0, 0, 0),
                                           average_radius=self.average_radius)
                        return self.generate_final_course(len(self.node_list)), i

        return None, i  # cannot find path

    def steer(self, parent_node, to_node, extend_length=float("inf")):
        """
        Create a child_node,
        calculate distance and angles of to_node from parent_node,
        find intermittent steps on the path towards the and save them for the child_node,
        return the child_node
        """

        # initiate child node at the parents position
        child_node = Node(parent_node.x, parent_node.y, parent_node.theta)

        # every node has a parent node, to whom he is connected
        child_node.parent = parent_node

        # polar angle is in conf_space stretched by average radius
        r, azimuth, polar = child_node.parent.calc_distance_and_angles(to_node, self.average_radius)

        child_node.path_x = [child_node.x]
        child_node.path_y = [child_node.y]
        child_node.path_theta = [child_node.theta]

        if extend_length > r:
            extend_length = r  # if the random point is closer than the step size, we alter our step size

        n_expand = math.floor(extend_length / self.path_resolution)  # number of points we check during our step

        # the child walks away from its parent node towards the to_node
        for _ in range(n_expand):
            # calculate the position of the new node in incremental steps
            child_node.x += self.path_resolution * math.cos(azimuth) * math.sin(polar)
            child_node.y += self.path_resolution * math.sin(azimuth) * math.sin(polar)
            child_node.theta = (child_node.theta + self.path_resolution * math.cos(polar)) % (2 * np.pi)
            # child_node.theta = (child_node.theta + self.path_resolution * math.cos(polar) + np.pi) % (2 * np.pi) - np.pi

            # list of all the points we passed over during our step
            child_node.path_x.append(child_node.x)
            child_node.path_y.append(child_node.y)
            child_node.path_theta.append(child_node.theta)

        # if we got very close to our sampled node (to node)
        r, _, _ = child_node.calc_distance_and_angles(to_node, self.average_radius)
        if r <= self.path_resolution:
            child_node.path_x.append(to_node.x)
            child_node.path_y.append(to_node.y)
            child_node.path_theta.append(to_node.theta)

            child_node.x = to_node.x
            child_node.y = to_node.y
            child_node.theta = to_node.theta

        return child_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.theta]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.theta % (2 * np.pi)])
            node = node.parent
        path.append([node.x, node.y, node.theta])
        path.append([self.end.x, self.end.y, self.end.theta])

        return np.array(path)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(*[random.uniform(dim[0], dim[1]) for dim in self.sampling_area])
        else:  # goal point sampling
            rnd = Node(self.end.x, self.end.y, self.end.theta)
        return rnd

    def plot_initial_conf_space(self):  # pragma: no cover
        fig = self.conf_space.visualize_space()
        self.start.draw_node(fig=fig, scale_factor=0.5, color=(0, 0, 0), average_radius=self.average_radius)
        self.end.draw_node(fig=fig, scale_factor=0.5, color=(0, 0, 0), average_radius=self.average_radius)
        return fig

    def collision_on_path(self, node):
        """
        Iterates over the points on the path to node,
        and checks at every point, whether there is a collision.
        return a boolean (True = collision, False = no collision)
        """
        for i in range(len(node.path_x)):
            if self.collision(node.path_x[i],
                              node.path_y[i],
                              node.path_theta[i]):
                return True  # collision
        return False  # no collision

    def collision(self, x, y, theta):
        """
        finds the indices_to_coords of (x, y, theta) in conf_space,
        where angles go from (0 to 2pi)
        """
        ind = self.conf_space.coords_to_indices(x, y, theta)
        if self.conf_space.space[ind]:  # if there is a 1, we collide
            return True  # collision
        return False


def main(size='XL', shape='SPT', solver='ant'):
    print("start " + __file__)

    # ====Search Path with RRT====
    conf_space = PhaseSpace.PhaseSpace(solver, size, shape,
                                       name=size + '_' + shape)
    conf_space.load_space()

    sampling_area = [[start[solver][shape][0] - 10, conf_space.extent['x'][-1] - 0.1],
                     [0, conf_space.extent['y'][-1]-0.1],
                     [0, 2 * np.pi]]

    # # large H
    # sampling_area = [[0, maze.slits[-1] + 5],
    #                  [0, maze.arena_height],
    #                  [0, 2 * np.pi]]

    # Set Initial parameters
    rrt = RRT(
        start=start[solver][shape],
        end=end[solver][shape],
        average_radius=average_radius(size, shape, solver,),
        sampling_area=sampling_area,
        conf_space=conf_space,
        path_resolution=conf_space.pos_resolution
    )

    fig = rrt.plot_initial_conf_space()
    path, it = rrt.planning(animation=show_animation,
                            fig=fig
                            )

    if path is None:
        print("Cannot find path")
    else:
        print("found path in {} iterations!!".format(str(it)))

        # Draw final path
        if show_animation:
            conf_space.draw(fig, path[:, 0:2], path[:, 2], scale_factor=0.4, color=(0, 1, 0))
            mlab.savefig(graph_dir() + os.path.sep + conf_space.name + '.jpg', magnification=4)
            mlab.close()
    return


if __name__ == '__main__':
    main()
