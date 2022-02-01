
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
import numpy as np
from os import path
import pickle
from Directories import SaverDirectories, work_dir, mini_SaverDirectories
from copy import deepcopy
from Setup.Maze import Maze
from Setup.Load import periodicity
from PhysicsEngine.Display import Display
from scipy.signal import savgol_filter
from Analysis.Velocity import velocity
from trajectory_inheritance.exp_types import is_exp_valid
from copy import copy

""" Making Directory Structure """
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['Small Far', 'Small Near', 'Medium', 'Large'],
         'humanhand': ''}

solvers = ['ant', 'human', 'humanhand', 'ps_simulation']

length_unit = {'ant': 'cm', 'human': 'm',  'humanhand': 'cm', 'ps_simulation': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


class Trajectory:
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, VideoChain=None):
        is_exp_valid(shape, solver, size)
        self.shape = shape  # shape (maybe this will become name of the maze...) (H, I, T, SPT)
        self.size = size  # size (XL, SL, L, M, S, XS)
        self.solver = solver  # ant, human, sim, humanhand
        self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        if VideoChain is None:
            self.VideoChain = [self.filename]
        else:
            self.VideoChain = VideoChain
        self.fps = fps  # frames per second
        self.position = np.empty((1, 2), float)  # np.array of x and y positions of the centroid of the shape
        self.angle = np.empty((1, 1), float)  # np.array of angles while the shape is moving
        self.frames = np.empty(0, float)
        self.winner = winner  # whether the shape crossed the exit
        self.participants = None

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def step(self, my_maze, i, display=None):
        my_maze.set_configuration(self.position[i], self.angle[i])

    def smooth(self):
        self.position[:, 0] = savgol_filter(self.position[:, 0], self.fps+1, 3)
        self.position[:, 1] = savgol_filter(self.position[:, 1], self.fps+1, 3)
        self.angle = savgol_filter(np.unwrap(self.angle), self.fps+1, 3) % (2 * np.pi)

    def interpolate_over_NaN(self):
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            nan_frames = np.unique(np.append(np.where(np.isnan(self.position))[0], np.where(np.isnan(self.angle))[0]))

            fr = [[nan_frames[0]]]
            for i in range(len(nan_frames) - 1):
                if abs(nan_frames[i] - nan_frames[i + 1]) > 1:
                    fr[-1] = fr[-1] + [nan_frames[i]]
                    fr = fr + [[nan_frames[i + 1]]]
            fr[-1] = fr[-1] + [nan_frames[-1]]
            print('Was NaN...' + str([self.frames[i].tolist() for i in fr]))

        # Some of the files contain NaN values, which mess up the Loading.. lets interpolate over them
        if np.any(np.isnan(self.position)) or np.any(np.isnan(self.angle)):
            for indices in fr:
                if indices[0] < 1:
                    indices[0] = 1
                if indices[1] > self.position.shape[0] - 2:
                    indices[1] = indices[1] - 1
                con_frames = indices[1] - indices[0] + 2
                self.position[indices[0] - 1: indices[1] + 1, :] = np.transpose(np.array(
                    [np.linspace(self.position[indices[0] - 1][0], self.position[indices[1] + 1][0], num=con_frames),
                     np.linspace(self.position[indices[0] - 1][1], self.position[indices[1] + 1][1], num=con_frames)]))
                self.angle[indices[0] - 1: indices[1] + 1] = np.squeeze(np.transpose(
                    np.array([np.linspace(self.angle[indices[0] - 1], self.angle[indices[1] + 1], num=con_frames)])))

    def divide_into_parts(self) -> list:
        """
        In order to treat the connections different than the actually tracked part, this function will split a single
        trajectory object into multiple trajectory objects.
        :return:
        """
        frame_dividers = [-1] + \
                         [i for i, (f1, f2) in enumerate(zip(self.frames, self.frames[1:])) if not f1 == f2-1] + \
                         [len(self.frames)]

        if len(frame_dividers)-1 != len(self.VideoChain):
            raise Exception('Why are your frames not matching your VideoChain in ' + self.filename + ' ?')

        parts = [Trajectory_part(self, [chain_element], [fr1+1, fr2+1])
                 for chain_element, fr1, fr2 in zip(self.VideoChain, frame_dividers, frame_dividers[1:])]
        return parts

    def timer(self):
        """

        :return: time in seconds
        """
        return (len(self.frames) - 1) / self.fps

    def iterate_coords(self, step=1) -> iter:
        """
        Iterator over (x, y, theta) of the trajectory
        :return: tuple (x, y, theta) of the trajectory
        """
        for pos, angle in zip(self.position[::step, :], self.angle[::step]):
            yield pos[0], pos[1], angle

    def find_contact(self):
        from PhysicsEngine.Contact import contact_loop_experiment
        my_maze = Maze(self)
        my_load = my_maze.bodies[-1]
        contact = []

        i = 0
        while i < len(self.frames):
            self.step(my_maze, i)  # update_screen the position of the load (very simple function, take a look)
            contact.append(contact_loop_experiment(my_load, my_maze))
            i += 1
        return contact

    def has_forcemeter(self):
        return False

    def old_filenames(self, i: int):
        if i > 0:
            raise Exception('only one old filename available')
        return self.filename

    def velocity(self, second_smooth, *args):
        return velocity(self.position, self.angle, self.fps, self.size, self.shape, second_smooth, self.solver, *args)

    def play(self, wait=0, ps=None, step=1, videowriter=False, frames=None):
        """
        Displays a given trajectory_inheritance (self)
        :param videowriter:
        :param frames:
        :param indices: which slice of frames would you like to display
        :param wait: how many milliseconds should we wait between displaying steps
        :param ps: Configuration space
        :param step: display only the ith frame
        :Keyword Arguments:
            * *indices_to_coords* (``[int, int]``) --
              starting and ending frame of trajectory_inheritance, which you would like to display
        """
        x = deepcopy(self)

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if frames is None:
            f1, f2 = 0, -1
        else:
            f1, f2 = frames[0], frames[1]
        x.position, x.angle = x.position[f1:f2:step, :], x.angle[f1:f2:step]
        x.frames = x.frames[f1:f2:step]

        if hasattr(x, 'participants') and x.participants is not None:   # TODO: this is a bit ugly, why does Amirs
            # have participants?
            x.participants.positions = x.participants.positions[f1:f2:step, :]
            x.participants.angles = x.participants.angles[f1:f2:step]
            if hasattr(x.participants, 'forces'):
                x.participants.forces.abs_values = x.participants.forces.abs_values[f1:f2:step, :]
                x.participants.forces.angles = x.participants.forces.angles[f1:f2:step, :]

        my_maze = Maze(x)
        return x.run_trj(my_maze, display=Display(x.filename, my_maze, wait=wait, ps=ps, videowriter=videowriter))

    def check(self):
        if self.frames.shape != self.angle.shape:
            raise Exception('Your frame shape does not match your angle shape!')

    def cut_off(self, frames: list):
        """

        :param frames: frame indices (not the yellow numbers on top)
        :return:
        """
        new = copy(self)
        new.frames = self.frames[frames[0]:frames[1]]
        new.position = self.position[frames[0]:frames[1]]
        new.angle = self.angle[frames[0]:frames[1]]
        return new

    def interpolate(self, frames_list: list):
        """

        :param frames_list: list of lists of frame indices (not the yellow numbers on top)
        :return:
        """
        from Load_tracked_data.Load_Experiment import connector
        new = copy(self)
        for frames in frames_list:
            con = connector(new.cut_off([0, frames[0]]), new.cut_off([frames[1], -1]), frames[1]-frames[0],
                            filename=self.filename + '_interpolation_' + str(frames[0]) + '_' + str(frames[1]))
            if frames[1]-frames[0] != con.position.shape[0]:
                extra_position = np.vstack([con.position[-1] for _ in range(frames[1]-frames[0] - con.position.shape[0])])
                extra_angle = np.hstack([con.angle[-1] for _ in range(frames[1]-frames[0] - con.angle.shape[0])])
                new.position[frames[0]:frames[1]] = np.vstack([con.position, extra_position])
                new.angle[frames[0]:frames[1]] = np.hstack([con.angle, extra_angle])
            else:
                new.position[frames[0]:frames[1]] = con.position
                new.angle[frames[0]:frames[1]] = con.angle  # TODO: why do I need squeeze here?
        return new

    def easy_interpolate(self, frames_list: list):
        """

        :param frames_list: list of lists of frame indices (not the yellow numbers on top)
        :return:
        """
        new = copy(self)
        for frames in frames_list:
            new.position[frames[0]:frames[1]] = np.vstack([new.position[frames[0]] for _ in range(frames[1]-frames[0])])
            new.angle[frames[0]:frames[1]] = np.hstack([new.angle[frames[0]] for _ in range(frames[1]-frames[0])])
        return new

    def geometry(self):
        pass

    def save(self, address=None) -> None:
        """
        1. save a pickle of the object
        2. save a pickle of a tuple of attributes of the object, in case I make a mistake one day, and change attributes
        in the class and then am incapable of unpickling my files.
        """
        self.check()
        if address is None:
            address = SaverDirectories[self.solver] + path.sep + self.filename

        with open(address, 'wb') as f:
            try:
                self_copy = deepcopy(self)
                if hasattr(self_copy, 'participants'):
                    delattr(self_copy, 'participants')
                pickle.dump(self_copy, f)
                print('Saving ' + self_copy.filename + ' in ' + address)
            except pickle.PicklingError as e:
                print(e)

        print('Saving minimal' + self.filename + ' in path: ' + mini_SaverDirectories[self.solver])
        pickle.dump((self.shape, self.size, self.solver, self.filename, self.fps,
                     self.position, self.angle, self.frames, self.winner),
                    open(mini_SaverDirectories[self.solver] + path.sep + self.filename, 'wb'))

    def stretch(self, frame_number: int) -> None:
        """
        I have to interpolate a trajectory. I know the frame number and a few points, that the shape should walk
        through.
        I have to stretch the path to these points over the given number of frames.
        :param frame_number:
        :return:
        """
        discont = np.pi / periodicity[self.shape]
        self.angle = np.unwrap(self.angle, discont=discont)
        stretch_factor = int(np.floor(frame_number/len(self.frames)))

        stretched_position = []
        stretched_angle = []
        if len(self.frames) == 1:
            stretched_position = np.vstack([self.position for _ in range(frame_number)])
            stretched_angle = np.vstack([self.angle for _ in range(frame_number)]).squeeze()

        for i, frame in enumerate(range(len(self.frames) - 1)):
            stretched_position += np.linspace(self.position[i], self.position[i+1], stretch_factor, endpoint=False).tolist()
            stretched_angle += np.linspace(self.angle[i], self.angle[i+1], stretch_factor, endpoint=False).tolist()

        self.position, self.angle = np.array(stretched_position), np.array(stretched_angle).squeeze()
        self.frames = np.array([i for i in range(self.angle.shape[0])])
        return

    def load_participants(self):
        pass

    def averageCarrierNumber(self):
        pass

    def run_trj(self, my_maze, display=None):
        i = 0
        while i < len(self.frames) - 1:
            self.step(my_maze, i, display=display)
            i += 1
            if display is not None:
                end = display.update_screen(self, i)
                if end:
                    display.end_screen()
                    self.frames = self.frames[:i]
                    break
                display.renew_screen(frame=self.frames[i], movie_name=self.filename)
        if display is not None:
            display.end_screen()

    def initial_cond(self):
        """
        We changed the initial condition. First, we had the SPT start between the two slits.
        Later we made it start in the back of the room.
        :return: str 'back' or 'front' depending on where the shape started
        """
        if self.shape != 'SPT':
            return None
        elif self.position[0, 0] < Maze(self).slits[0]:
            return 'back'
        return 'front'

    def communication(self):
        return False


class Trajectory_part(Trajectory):
    def __init__(self, parent_traj, VideoChain: list, frames: list):
        """

        :param parent_traj:
        :param VideoChain:
        :param frames: []
        """
        super().__init__(size=parent_traj.size, shape=parent_traj.shape, solver=parent_traj.solver,
                         filename=parent_traj.filename, fps=parent_traj.fps, winner=parent_traj.winner,
                         VideoChain=VideoChain)
        self.parent_traj = parent_traj
        self.frames_of_parent = frames
        self.frames = parent_traj.frames[frames[0]:frames[-1]]
        self.position = parent_traj.position[frames[0]:frames[-1]]
        self.angle = parent_traj.angle[frames[0]:frames[-1]]

    def is_connector(self):
        return 'CONNECTOR' in self.VideoChain[-1]

    def geometry(self):
        return self.parent_traj.geometry()


def get(filename) -> Trajectory:
    import os
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            if filename in os.listdir(os.path.join(work_dir, dir)):
                address = os.path.join(work_dir, dir, filename)
                with open(address, 'rb') as f:
                    x = pickle.load(f)
                return x
    else:
        raise ValueError('I cannot find ' + filename)