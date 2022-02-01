import numpy as np
import pickle
from PhysicsEngine.Contact import possible_configuration
import os
import itertools
from Setup.Maze import Maze
from Directories import PhaseSpaceDirectory
from Analysis.PathLength import resolution
from scipy import ndimage
from datetime import datetime
import string
from skfmm import distance
from tqdm import tqdm
from Analysis.States import forbidden_transition_attempts, allowed_transition_attempts, states
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from copy import copy

try:
    from mayavi import mlab
except:
    print('mayavi not installed')

try:
    import cc3d
except:
    print('cc3d not installed')

traj_color = (1.0, 0.0, 0.0)
start_end_color = (0.0, 0.0, 0.0)
scale = 5


# TODO: fix the x.winner attribute

# I want the resolution (in cm) for x and y and archlength to be all the same.


class PhaseSpace(object):
    def __init__(self, solver: str, size: str, shape: str, geometry: tuple, name=""):
        """
        :param board_coords:
        :param load_coords:
        # :param pos_resolution: load replacement resolution (in in the coords units)
        # :param theta_resolution: theta replace
        # :param x_range: tuple of the x-space range, in the coords units
        # :param y_range: tuple of the y-space range, in the coords units
        """
        maze = Maze(size=size, shape=shape, solver=solver, geometry=geometry)

        if len(name) == 0:
            name = size + '_' + shape

        self.name = name
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry: tuple = geometry

        x_range = (0, maze.slits[-1] + max(maze.getLoadDim()) + 1)
        y_range = (0, maze.arena_height)

        self.extent = {'x': x_range,
                       'y': y_range,
                       'theta': (0, np.pi * 2)}
        self.average_radius = maze.average_radius()

        self.pos_resolution = self.extent['y'][1] / self.number_of_points()['y']
        self.theta_resolution = 2 * np.pi / self.number_of_points()['theta']

        self.space = None  # True, if configuration is possible; False, if there is a collision with the wall
        self.space_boundary = None
        self.fig = None

        # self.monitor = {'left': 3800, 'top': 160, 'width': 800, 'height': 800}
        # self.VideoWriter = cv2.VideoWriter('mayavi_Capture.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 20,
        #                                    (self.monitor['width'], self.monitor['height']))
        # self._initialize_maze_edges()

    def directory(self, point_particle: bool = False, erosion_radius: int = None, addition: str = '', small=False):
        """
        :param addition:
        :param erosion_radius:
        :param point_particle:
        :param small: if you want only the labeled configuration space
        where the phase space is saved
        If an erosion_radius is given, we are dealing with a labeled Phase Space.
        """

        if self.size in ['Small Far', 'Small Near']:  # both have same dimensions
            size = 'Small'
        else:
            size = self.size

        filename = size + '_' + self.shape + '_' + self.geometry[0][:-5]

        if point_particle:
            return os.path.join(PhaseSpaceDirectory, self.solver, self.shape, filename + addition + '_pp.pkl')
        if erosion_radius is not None:
            path_ = os.path.join(PhaseSpaceDirectory, self.solver, self.shape, filename + '_labeled_erosion_'
                                 + str(erosion_radius) + addition + '.pkl')
            if small:
                path_ = path_[:-4] + '_small' + '.pkl'
            return path_

        path_ = os.path.join(PhaseSpaceDirectory, self.solver, self.shape, filename + addition + '.pkl')
        if small:
            path_ = path_[:-4] + '_small' + '.pkl'
        return path_

    def number_of_points(self) -> dict:
        # x_num = np.ceil(self.extent['x'][1]/resolution)
        res = resolution(self.size, self.solver, self.shape)
        y_num = np.ceil(self.extent['y'][1] / res)
        theta_num = np.ceil(self.extent['theta'][1] * self.average_radius / res)
        return {'x': None, 'y': y_num, 'theta': theta_num}

    def initialize_maze_edges(self) -> None:
        """
        set x&y edges to 0 in order to define the maze boundaries (helps the visualization)
        :return:
        """
        self.space[0, :, :] = False
        self.space[-1, :, :] = False
        self.space[:, 0, :] = False
        self.space[:, -1, :] = False

    def calculate_space(self, point_particle=False, mask=None) -> None:
        # TODO: implement point particles
        maze = Maze(size=self.size, shape=self.shape, solver=self.solver, geometry=self.geometry)
        load = maze.bodies[-1]

        # initialize 3d map for the phase_space
        self.space = self.empty_space()
        print("PhaseSpace: Calculating space " + self.name)

        # how to iterate over phase space
        def ps_calc():
            """
            param x0: index of x array to start with
            param x1: index of x array to end with
            :return: iterator"""
            maze_corners = np.array_split(maze.corners(), int(maze.corners().shape[0] / 4))
            former_found = (0, 0)
            for x, y, theta in self.iterate_coordinates(mask=mask):
                indices = self.coords_to_indices(x, y, theta)
                maze.set_configuration([x, y], float(theta))
                self.space[indices], former_found = possible_configuration(load, maze_corners, former_found)
                # load.position, load.angle = [x, y], float(theta)
                # from PhysicsEngine.Display import Display
                # display = Display('', maze)
                # maze.draw(display)
                # display.display()
        ps_calc()

    def new_fig(self):
        fig = mlab.figure(figure=self.name, bgcolor=(1, 1, 1,), fgcolor=(0, 0, 0,), size=(800, 800))
        return fig

    @staticmethod
    def reduced_resolution(space: np.array, reduction: int):
        """
        Reduce the resolution of the PS.
        :param space: space that you want to reshpe
        :param reduction: By what factor the PS will be reduced.
        :return:
        """
        for axis in range(int(space.ndim)):
            space = space.take(indices=range(0, int(space.shape[axis] / reduction) * reduction), axis=axis)

        def reshape(array) -> np.array:
            """
            Shrink an array
            :return:
            """
            reshaper = [item for t in [(int(axis_len / reduction), reduction)
                                       for axis_len in array.shape] for item in t]

            return array.reshape(*reshaper)

        def summer(array) -> np.array:
            for i in range(int(array.ndim / 2)):
                array = array.sum(axis=i + 1)
            return array

        # return np.array(summer(reshape(space))/(reduction**space.ndim)>0.5, dtype=bool)
        return summer(reshape(space)) / (reduction ** space.ndim)

    def visualize_space(self, reduction=1, fig=None, colormap='Greys', space: np.ndarray =None) -> None:
        if fig is None and (self.fig is None or not self.fig.running):
            self.fig = self.new_fig()
        else:
            self.fig = fig

        x, y, theta = np.mgrid[self.extent['x'][0]:self.extent['x'][1]:self.pos_resolution * reduction,
                      self.extent['y'][0]:self.extent['y'][1]:self.pos_resolution * reduction,
                      self.extent['theta'][0]:self.extent['theta'][1]:self.theta_resolution * reduction,
                      ]

        # os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (800, 160)
        if self.space is None:
            self.load_space()

        if space is None:
            space = np.array(self.space, dtype=int)
        else:
            space = np.array(space, dtype=int)

        if reduction > 1:
            space = self.reduced_resolution(space, reduction)

        def prune(array1, array2):
            for axis in range(array1.ndim):
                if array2.shape[axis] > array1.shape[axis]:
                    array2 = array2.take(indices=range(array1.shape[axis]), axis=axis)
                elif array2.shape[axis] < array1.shape[axis]:
                    array1 = array1.take(indices=range(array2.shape[axis]), axis=axis)
            return array1, array2

        if x.shape != space.shape:
            x, space = prune(x, space)
            y, space = prune(y, space)
            theta, space = prune(theta, space)

        cont = mlab.contour3d(x, y, theta,
                              space[:x.shape[0], :x.shape[1], :x.shape[2]],
                              opacity=0.08,  # 0.15
                              figure=self.fig,
                              colormap=colormap)

        cont.actor.actor.scale = [1, 1, self.average_radius]
        mlab.view(-90, 90)

    def iterate_space_index(self, mask) -> iter:
        """
        :return: iterator over the indices_to_coords of self.space
        """
        if mask is None:
            for x_i in tqdm(range(self.space.shape[0])):
                for y_i, theta_i in itertools.product(range(self.space.shape[1]),
                                                      range(self.space.shape[2])):
                    yield x_i, y_i, theta_i
        else:
            for x_i in tqdm(tqdm(range(np.array(np.where(mask))[:, 0][0], np.array(np.where(mask))[:, -1][0]))):
                for y_i, theta_i in itertools.product(range(self.space.shape[1]),
                                                      range(self.space.shape[2])):
                    yield x_i, y_i, theta_i

    def iterate_coordinates(self, mask=None) -> iter:
        r"""
        param x0: index to start with
        param x1: index to end with
        :return: iterator
        """
        if mask is None:
            x_iter = np.arange(self.extent['x'][0], self.extent['x'][1], self.pos_resolution)
            for x in tqdm(x_iter):
                for y in np.arange(self.extent['y'][0], self.extent['y'][1], self.pos_resolution):
                    for theta in np.arange(self.extent['theta'][0], self.extent['theta'][1], self.theta_resolution):
                        yield x, y, theta
        else:
            x0, y0, theta0 = np.array(np.where(mask))[:, 0]
            x1, y1, theta1 = np.array(np.where(mask))[:, -1]
            for x in tqdm(np.arange(self.indices_to_coords(x0, 0, 0)[0],
                                    self.indices_to_coords(x1, 0, 0)[0],
                                    self.pos_resolution)):
                for y in np.arange(self.indices_to_coords(0, y0, 0)[1],
                                   self.indices_to_coords(0, y1, 0)[1],
                                   self.pos_resolution):
                    for theta in np.arange(self.indices_to_coords(0, 0, theta0)[2],
                                           self.indices_to_coords(0, 0, theta1)[2],
                                           self.theta_resolution):
                        if mask[self.coords_to_indices(x, y, theta)]:
                            yield x, y, theta

    def iterate_neighbours(self, ix, iy, itheta) -> iter:
        for dx, dy, dtheta in itertools.product([-1, 0, 1], repeat=3):
            _ix, _iy, _itheta = ix + dx, iy + dx, itheta + dtheta
            if ((_ix >= self.space.shape[0]) or (_ix < 0)
                    or (_iy >= self.space.shape[1]) or (_iy < 0)
                    or (_itheta >= self.space.shape[2]) or (_itheta < 0)
                    or ((dx, dy, dtheta) == (0, 0, 0))):
                continue
            yield _ix, _iy, _itheta

    # @staticmethod
    # def plot_trajectory(traj, color=(0, 0, 0)):
    #     mlab.plot3d(traj[0],
    #                 traj[1],
    #                 traj[2],
    #                 color=color, tube_radius=0.045, colormap='Spectral')
    #     mlab.points3d([traj[0, 0]], [traj[1, 0]], [traj[2, 0]])

    def save_space(self, directory=None):
        """
        Pickle the numpy array in given path, or in default path. If default directory exists, add a string for time, in
        order not to overwrite the old .pkl file.
        :param path: Where you would like to save.
        :return:
        """
        if not hasattr(self, 'space'):
            self.calculate_space()
        if not hasattr(self, 'space_boundary'):
            self.calculate_boundary()
        if directory is None:
            if os.path.exists(self.directory()):
                now = datetime.now()
                date_string = '_' + now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
                directory = self.directory(addition=date_string)
        print('Saving ' + self.name + ' in path: ' + directory)
        pickle.dump((np.array(self.space, dtype=bool),
                     np.array(self.space_boundary, dtype=bool),
                     self.extent),
                    open(directory, 'wb'))

    def load_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        """
        directory = self.directory(point_particle=point_particle)
        if os.path.exists(directory):
            (self.space, self.space_boundary, self.extent) = pickle.load(open(directory, 'rb'))
            self.initialize_maze_edges()
            if self.extent['theta'] != (0, 2 * np.pi):
                print('need to correct' + self.name)
        else:
            self.calculate_boundary(point_particle=point_particle)
            self.save_space()
        return

    def _is_boundary_cell(self, x, y, theta) -> bool:
        if not self.space[x, y, theta]:
            return False
        for n_x, n_y, n_theta in self.iterate_neighbours(x, y, theta):
            if not self.space[n_x, n_y, n_theta]:
                return True
        return False

    def indices_to_coords(self, ix, iy, itheta) -> tuple:
        return (self.extent['x'][0] + ix * self.pos_resolution,
                self.extent['y'][0] + iy * self.pos_resolution,
                self.extent['theta'][0] + itheta * self.theta_resolution)

    def coords_to_index(self, axis: int, value):
        if value is None:
            return None
        resolut = {0: self.pos_resolution, 1: self.pos_resolution, 2: self.theta_resolution}[axis]
        value_i = min(int(np.round((value - list(self.extent.values())[axis][0]) / resolut)),
                      self.space.shape[axis] - 1)

        if value_i >= self.space.shape[axis] or value_i <= -1:
            print('check', list(self.extent.keys())[axis])
        return value_i

    def coords_to_indices(self, x: float, y: float, theta: float) -> tuple:
        """
        convert coordinates into indices_to_coords in PhaseSpace
        :param x: x position of CM in cm
        :param y: y position of CM in cm
        :param theta: orientation of axis in radian
        :return: (xi, yi, thetai)
        """
        return self.coords_to_index(0, x), self.coords_to_index(1, y), \
               self.coords_to_index(2, theta % (2 * np.pi))

    def empty_space(self) -> np.array:
        return np.zeros((int(np.ceil((self.extent['x'][1] - self.extent['x'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['y'][1] - self.extent['y'][0]) / float(self.pos_resolution))),
             int(np.ceil((self.extent['theta'][1] - self.extent['theta'][0]) / float(self.theta_resolution)))),
                        dtype=bool)

    def calculate_boundary(self, point_particle=False, mask=None) -> None:
        """

        :param point_particle:
        :param mask: Where to calculate space (usefull just for testing)
        :return:
        """
        if self.space is None:
            self.calculate_space(point_particle=point_particle, mask=mask)
        self.space_boundary = self.empty_space()
        print("PhaseSpace: Calculating boundaries " + self.name)
        for ix, iy, itheta in self.iterate_space_index(mask=mask):
            if self._is_boundary_cell(ix, iy, itheta):
                self.space_boundary[ix, iy, itheta] = 1

    def draw(self, positions, angles, scale_factor: float = 0.5, color=(1, 0, 0)) -> None:
        """
        draw positions and angles in 3 dimensional phase space
        """
        if np.array(positions).ndim == 1:
            positions = np.expand_dims(np.array(positions), axis=0)
            angles = np.expand_dims(np.array(angles), axis=0)
        # if self.VideoWriter is None:
        #     self.VideoWriter = cv2.VideoWriter('mayavi_Capture.mp4v', cv2.VideoWriter_fourcc(*'DIVX'), 20, (354, 400))
        angle = angles * self.average_radius
        mlab.points3d(positions[:, 0], positions[:, 1], angle,
                      figure=self.fig,
                      scale_factor=scale_factor,
                      color=color
                      )

    def draw_ind(self, indices, scale_factor=0.2):
        coords = self.indices_to_coords(*indices)
        self.draw(coords[:2], coords[2], scale_factor=scale_factor)

    def trim(self, borders) -> None:
        [[x_min, x_max], [y_min, y_max]] = borders
        self.extent['x'] = (max(0, x_min), min(self.extent['x'][1], x_max))
        self.extent['y'] = (max(0, y_min), min(self.extent['y'][1], y_max))
        self.space = self.space[max(0, int(x_min / self.pos_resolution)):
                                min(int(x_max / self.pos_resolution) + 1, self.space.shape[0]),
                     max(0, int(y_min / self.pos_resolution)):
                     min(int(y_max / self.pos_resolution) + 1, self.space.shape[1]),
                     ]

    @staticmethod
    def dilate(space: np.array, radius: int) -> np.array:
        """
        dilate phase space
        :param radius: radius of dilation
        """
        print('Dilating space...')
        struct = np.ones([radius for _ in range(space.ndim)], dtype=bool)
        return np.array(ndimage.binary_dilation(space, structure=struct), dtype=bool)

    @staticmethod
    def erode(space, radius: int) -> np.array:
        """
        Erode phase space.
        We erode twice
        :param space: Actual space you want to erode
        :param radius: radius of erosion
        """
        print('Eroding space...')

        def erode_space(space, struct):
            return ndimage.binary_erosion(space, structure=struct)
            # return np.array(~ndimage.binary_erosion(~np.array(space, dtype=bool), structure=struct), dtype=bool)

        struct = np.ones([radius for _ in range(space.ndim)], dtype=bool)

        space1 = erode_space(space, struct)

        slice = int(space.shape[-1] / 2)
        space2 = erode_space(np.concatenate([space[:, :, slice:], space[:, :, :slice]], axis=2), struct)
        space2 = np.concatenate([space2[:, :, slice:], space2[:, :, :slice]], axis=2)

        return np.logical_or(space1, space2)  # have to check, if its really a logical or...

    def split_connected_components(self, space: np.array, cc_to_keep=10) -> (list, list):
        """
        from self find connected components
        Take into account periodicity
        :param cc_to_keep: how many connected components to keep (10 for SPT)
        :param space: which space should be split
        :return: list of ps spaces, that have only single connected components
        """
        ps_states = []
        letters = list(string.ascii_lowercase)
        centroids = np.empty((0, 3))
        labels, number_cc = cc3d.connected_components(space, connectivity=6, return_N=True)
        stats = cc3d.statistics(labels)

        min = max(2000, np.sort([stats['voxel_counts'][label] for label in range(1, number_cc)])[-cc_to_keep]-1)

        for label in range(1, number_cc):
            if stats['voxel_counts'][label] > min:
                ps = PS_Area(self, np.bool_(labels == label), letters.pop(0))

                # if this is part of a another ps that is split by 0 or 2pi
                centroid = np.array(self.indices_to_coords(*np.floor(stats['centroids'][label])))

                border_bottom = np.any(ps.space[:, :, 0])
                border_top = np.any(ps.space[:, :, -1])

                if (border_bottom and not border_top) or (border_top and not border_bottom):
                    index = np.where(np.abs(centroid[0] - centroids[:, 0]) < 0.1)[0]
                    if len(index) > 0:
                        ps_states[index[0]].space = np.logical_or(ps_states[index[0]].space, ps.space)
                        centroids[index[0]][-1] = 0

                    else:
                        ps_states.append(ps)
                        centroids = np.vstack([centroids, centroid])
                else:
                    ps_states.append(ps)
                    centroids = np.vstack([centroids, centroid])
        return ps_states, centroids

    def overlapping(self, ps_area):
        return np.any(self.space[ps_area.space])


class PS_Area(PhaseSpace):
    def __init__(self, ps: PhaseSpace, space: np.array, name: str):
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape, geometry=ps.geometry)
        self.space: np.array = space
        self.fig = ps.fig
        self.name: str = name
        self.distance: np.array = None

    def calculate_distance(self, mask: np.array) -> np.array:
        """
        Calculate the distance to every other node in the array.
        :param mask: Distances have to be calculated according to the available nodes.
        The available nodes are saved in 'mask'
        :return: np.array with distances from each node
        """

        # self.distance = distance(np.array((~np.array(self.space, dtype=bool)), dtype=int), periodic=(0, 0, 1))
        phi = np.array(~self.space, dtype=int)
        masked_phi = np.ma.MaskedArray(phi, mask=mask)
        self.distance = distance(masked_phi, periodic=(0, 0, 1))

        # node at (105, 36, 102)

        # point = (105, 36, 102)
        # self.draw(self.indices_to_coords(*point)[:2], self.indices_to_coords(*point)[-1])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[point[0], :, :])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[:, point[1], :])
        # plt.imshow(distance(masked_phi, periodic=(0, 0, 1))[:, :, point[2]])


class PS_Mask(PS_Area):
    def __init__(self, ps):
        space = np.zeros_like(ps.space, dtype=bool)
        super().__init__(ps, space, 'mask')

    @staticmethod
    def paste(wall: np.array, block: np.array, loc: tuple) -> np.array:
        """
        :param wall:
        :param block:
        :param loc: the location of the wall, that should be in the (0, 0) position of the block
        """

        def paste_slices(tup) -> tuple:
            pos, w, max_w = tup
            wall_min = max(pos, 0)
            wall_max = min(pos + w, max_w)
            block_min = -min(pos, 0)
            block_max = max_w - max(pos + w, max_w)
            block_max = block_max if block_max != 0 else None
            return slice(wall_min, wall_max), slice(block_min, block_max)

        loc_zip = zip(loc, block.shape, wall.shape)
        wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
        wall[wall_slices] = block[block_slices]
        return wall

    def add_circ_mask(self, radius: int, indices: tuple):
        """
        because of periodic boundary conditions in theta, we first roll the array, then assign, then unroll
        add a circular mask to
        """

        def circ_mask() -> np.array:
            x, y, theta = np.ogrid[(-1) * radius: radius + 1, (-1) * radius: radius + 1, (-1) * radius: radius + 1]
            mask = np.array(x ** 2 + y ** 2 + theta ** 2 <= radius ** 2)
            return mask

        loc = (-radius + indices[0], -radius + indices[1], 0)
        self.paste(self.space, circ_mask(), loc)
        self.space = np.roll(self.space, -radius + indices[2], axis=2)


class Node:
    def __init__(self, indices):
        self.indices: tuple = indices

    def draw(self, ps):
        ps.draw_ind(self.indices)

    def find_closest_state(self, ps_states: list) -> int:
        """
        :return: name of the ps_state closest to indices_to_coords, chosen from ps_states
        """
        for radius in range(1, 50):
            print(radius)
            ps_mask = PS_Mask(ps_states[0])
            ps_mask.add_circ_mask(radius, self.indices)

            for ps_state in ps_states:
                if ps_state.overlapping(ps_mask):
                    print(ps_state.name)
                    # self.visualize_space()
                    # ps_state.visualize_space(self.fig)
                    # ps_mask.visualize_space(self.fig, colormap='Oranges')
                    return ps_state.name
        return 0

    def find_closest_states(self, ps_states: list, N: int = 1) -> list:
        """
        :param N: how many of the closest states do you want to find?
        :param ps_states: how many of the closest states do you want to find?
        :return: name of the closest PhaseSpace
        """

        state_order = []  # carries the names of the closest states, from closest to farthest

        for i in range(N):
            closest = self.find_closest_state(ps_states)
            state_order.append(closest)
            [ps_states.remove(ps_state) for ps_state in ps_states if ps_state.name == closest]
        return state_order


class PhaseSpace_Labeled(PhaseSpace):
    """
    This class stores configuration space for a piano_movers problem in a 3 dim array.
    Axis 0 = x direction
    Axis 1 = y direction
    Axis 0 = x direction
    Every element of self.space carries on of the following indices_to_coords of:
    - '0' (not allowed)
    - A, B...  (in self.eroded_space), where N is the number of states
    - n_1 + n_2 where n_1 and n_2 are in (A, B...).
        n_1 describes the state you came from.
        n_2 describes the state you are
    """

    def __init__(self, solver, size, shape, geometry, ps: PhaseSpace = None):
        if ps is None:
            ps = PhaseSpace(solver, size, shape, geometry, name='')
            ps.load_space()
        super().__init__(solver=ps.solver, size=ps.size, shape=ps.shape, geometry=geometry)
        self.space = ps.space  # True, if there is collision. False, if it is an allowed configuration

        self.eroded_space = None
        self.erosion_radius = self.erosion_radius_default()
        self.ps_states = self.centroids = None
        self.space_labeled = None

    def load_eroded_labeled_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle. Load both eroded space, and ps_states, centroids and everything
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        """
        directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius)

        if os.path.exists(directory):
            print('Loading labeled from ', directory, '...')
            self.eroded_space, self.ps_states, self.centroids, self.space_labeled = pickle.load(open(directory, 'rb'))
        else:
            self.eroded_space = self.erode(self.space, radius=self.erosion_radius)
            self.ps_states, self.centroids = self.split_connected_components(self.eroded_space)
            # self.visualize_states(reduction=5)
            self.label_space()
            self.save_labeled()
        print('Finished loading')

    def load_labeled_space(self, point_particle: bool = False) -> None:
        """
        Load Phase Space pickle.
        :param point_particle: point_particles=True means that the load had no fixtures when ps was calculated.
        """
        directory = self.directory(point_particle=point_particle, erosion_radius=self.erosion_radius, small=True)

        print('Loading labeled from ', directory, '...')
        if os.path.exists(directory):
            # self.space_labeled = pickle.load(open(path, 'rb'))
            self.space_labeled = pickle.load(open(directory, 'rb'))
        else:
            self.load_eroded_labeled_space()

    def check_labels(self) -> list:
        """
        I want to check, whether there are labels, that I don't want to have.
        :return: list of all the labels that are not labels I wanted to have
        """
        return [label for label in np.unique(self.space_labeled) if label not in
                states + forbidden_transition_attempts + allowed_transition_attempts]

    # def label_space_slow(self) -> None:
    #     """
    #     label each node in PhaseSpace.space with a list
    #     """
    #     self.space_labeled = np.zeros([*self.space.shape, 2])
    #
    #     def calculate_label_slow(indices_to_coords: tuple) -> list:
    #         """
    #         Finds the label for the coordinates x, y and theta
    #         :return: integer or list
    #         """
    #         # everything not in self.space.
    #         if self.space[indices_to_coords]:
    #             return [0, np.NaN]
    #
    #         # everything in self.ps_states.
    #         for i, ps in enumerate(self.ps_states):
    #             # [not self.ps_states[i].space[indices_to_coords] for i in range(len(self.ps_states))]
    #             if ps.space[indices_to_coords]:
    #                 return [i, np.NaN]
    #
    #         # in eroded space
    #         return Node(indices_to_coords).find_closest_states(self.ps_states, N=2)
    #
    #     def label_slice(x_i):
    #         matrix = np.zeros((self.space.shape[1], self.space.shape[2], 2))
    #         for y_i, theta_i in itertools.product(range(self.space.shape[1]), range(self.space.shape[2])):
    #             matrix[y_i, theta_i] = calculate_label_slow((x_i, y_i, theta_i))
    #         print('finished slice ', str(x_i))
    #         return matrix
    #
    #     start_time = datetime.now()
    #     label_slice(100)
    #     print('Duration: {}'.format(datetime.now() - start_time))
    #
    #     matrices = Parallel(n_jobs=4)(delayed(label_slice)(x_i) for x_i in range(self.space.shape[0]))
    #     self.space_labeled = np.stack(matrices, axis=0)

    def visualize_states(self, fig=None, colormap='Oranges', reduction: int = 1) -> None:
        """

        :param fig: mylab figure reference
        :param colormap: What color do you want the available states to appear in?
        :param reduction: What amount of reduction?
        :return:
        """
        if self.fig is None or not self.fig.running:
            self.visualize_space(reduction=reduction)

        else:
            self.fig = fig

        print('Draw states')
        for centroid, ps_state in tqdm(zip(self.centroids, self.ps_states)):
            # ps_state.extent = self.extent # This was only becaus I had made a mistake
            ps_state.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction)
            mlab.text3d(*(centroid * [1, 1, self.average_radius]), ps_state.name, scale=2)

    def visualize_transitions(self, fig=None, reduction: int = 1) -> None:
        """

        :param fig: mylab figure reference
        :param colormap: What color do you want the available states to appear in?
        :param reduction: What amount of reduction?
        :return:
        """
        if self.fig is None or not self.fig.running:
            self.visualize_space(reduction=reduction)

        else:
            self.fig = fig

        print('Draw transitions')
        idx = 0
        for label in tqdm(np.unique(self.space_labeled)):
            if len(label) > 1:
                if idx % 3 == 0:
                    colormap = 'Reds'
                elif idx % 3 == 1:
                    colormap = 'Purples'
                else:
                    colormap = 'Greens'
                space = np.array(self.space_labeled == label, dtype=bool)
                centroid = self.indices_to_coords(*np.array(np.where(space))[:, 0])
                self.visualize_space(fig=self.fig, colormap=colormap, reduction=reduction, space=space)
                mlab.text3d(*(a*b for a, b in zip(centroid, [1, 1, self.average_radius])), label, scale=1)
                idx += 1

    def save_labeled(self, directory=None, date_string='') -> None:
        if directory is None:
            now = datetime.now()
            date_string = now.strftime("%Y") + '_' + now.strftime("%m") + '_' + now.strftime("%d")
            directory = self.directory(point_particle=False, erosion_radius=self.erosion_radius, addition=date_string)

        print('Saving ' + self.name + ' in path: ' + directory)
        pickle.dump((self.eroded_space, self.ps_states, self.centroids, self.space_labeled), open(path, 'wb'))

        # Actually, I dont really need all this information.  self.space_labeled should be enough
        directory = self.directory(point_particle=False,
                                   erosion_radius=self.erosion_radius, addition=date_string, small=True)

        print('Saving reduced in' + self.name + ' in path: ' + directory)
        pickle.dump(self.space_labeled, open(directory, 'wb'))
        print('Finished saving')

    def erosion_radius_default(self) -> int:
        """
        for ant XL, the right radius of erosion is 0.9cm. To find this length independent of scaling for every system,
        we use this function.
        :return:
        """
        return self.coords_to_indices(0, (self.extent['y'][1] / 21.3222222), 0)[1]
        # return int(np.ceil(self.coords_to_indices(0, 0.9, 0)[0]))

    def max_distance_for_transition(self):
        """
        :return: maximum distance so that it is noted as transition area
        """
        maze = Maze(solver=self.solver, size=self.size, shape=self.shape)
        if self.shape == 'SPT':
            distance_cm = (maze.slits[1] - maze.slits[0]) / 3
        else:
            distance_cm = maze.exit_size / 2
        return self.coords_to_index(0, distance_cm) + self.erosion_radius * 2

    def label_space(self) -> None:
        print('Calculating distances for the different states in', self.name)
        dilated_space = self.dilate(self.space, self.erosion_radius_default())
        [ps_state.calculate_distance(~dilated_space) for ps_state in tqdm(self.ps_states)]
        distance_stack_original = np.stack([ps_state.distance for ps_state in self.ps_states], axis=3)

        distance_stack = np.stack([ps_state.distance for ps_state in self.ps_states], axis=3)
        far_away = distance_stack > self.max_distance_for_transition()
        distance_stack[far_away] = np.inf

        ps_name_dict = {i: ps_state.name for i, ps_state in enumerate(self.ps_states)}

        def assign_label(ind: tuple):
            """
            Assign label to node in space.
            :param ind: indices of node in configuration space
            :return:
            """
            # everything not in self.space.
            if not self.space[ind]:
                self.space_labeled[ind] = '0'
                return

            # everything in self.ps_states.
            for i, ps_state in enumerate(self.ps_states):
                if ps_state.space[ind]:
                    self.space_labeled[ind] = ps_state.name
                    return

            # in eroded space
            # self.visualize_states()
            # self.draw_ind(indices)
            self.space_labeled[ind] = ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack[ind])[:2]
                                               if distance_stack[ind].data[ii] < np.inf])
            if len(self.space_labeled[ind]) == 0:
                self.space_labeled[ind] = ''.join([ps_name_dict[ii] for ii in np.argsort(distance_stack_original[ind])[:2]])

        self.space_labeled = np.zeros_like(self.space, dtype=np.dtype('U2'))
        print('Iterating over every node and assigning label')
        [assign_label(indices) for indices in self.iterate_space_index()]


if __name__ == '__main__':
    # shape = 'SPT'
    # size = 'M'
    # solver = 'ant'

    # ps = PhaseSpace(solver, size, shape)
    # ps.calculate_space()
    # ps.calculate_boundary()
    #
    # ps.save_space()
    # ps.visualize_space()

    DEBUG = 1
