import numpy as np
from Box2D import b2BodyDef, b2_staticBody, b2World, b2_dynamicBody, b2FixtureDef, b2CircleShape, b2Vec2
from Setup.MazeFunctions import BoxIt
from scipy.spatial import cKDTree
from pandas import read_excel
from Directories import maze_dimension_directory
from PhysicsEngine.drawables import Polygon, Point, Circle, colors
from copy import copy
from os import path
from trajectory_inheritance.exp_types import is_exp_valid

ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie']  # also in Maze.py

# TODO: x = get(myDataFrame.loc[429].filename).play() displays a maze, that does not make any sense!

periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}
ASSYMETRIC_H_SHIFT = 1.22 * 2
# SPT_RATIO = 2.44 / 4.82  # ratio between shorter and longer side on the Special T
centerOfMass_shift = - 0.08  # shift of the center of mass away from the center of the load. # careful!
# My PS are still for the original value below!!
# centerOfMass_shift = - 0.10880829015544041  # shift of the center of mass away from the center of the load.

size_per_shape = {'ant': {'H': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'I': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'T': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
                          'SPT': ['S', 'M', 'L', 'XL'],
                          'RASH': ['S', 'M', 'L', 'XL'],
                          'LASH': ['S', 'M', 'L', 'XL'],
                          },
                  'human': {'SPT': ['S', 'M', 'L']},
                  'humanhand': {'SPT': ['']}
                  }

StateNames = {'H': [0, 1, 2, 3, 4, 5], 'I': [0, 1, 2, 3, 4, 5], 'T': [0, 1, 2, 3, 4, 5],
              'SPT': [0, 1, 2, 3, 4, 5, 6], 'LASH': [0, 1, 2, 3, 4, 5, 6], 'RASH': [0, 1, 2, 3, 4, 5, 6],
              'circle': [0]}

ResizeFactors = {'ant': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'ps_simulation': {'XL': 1, 'SL': 0.75, 'L': 0.5, 'M': 0.25, 'S': 0.125, 'XS': 0.125 / 2},
                 'human': {'Small Near': 1, 'Small Far': 1, 'S': 1, 'M': 1, 'Medium': 1, 'Large': 1, 'L': 1},
                 'humanhand': {'': 1}}

for solver in ant_dimensions:
    ResizeFactors[solver] = ResizeFactors['ant']


# there are a few I mazes, which have a different exit size,

# x, y, theta
def start(size, shape, solver):
    maze = Maze(size=size, shape=shape, solver=solver)

    if shape == 'SPT':
        # return [(maze.slits[0] - maze.slits[-1]) / 2 + maze.slits[-1] - 0.5, maze.arena_height / 2, 0]
        return [maze.slits[0] * 0.5, maze.arena_height / 2, 0]
    elif shape in ['H', 'I', 'T', 'RASH', 'LASH']:
        return [maze.slits[0] - 5, maze.arena_height / 2, np.pi - 0.1]


def end(size, shape, solver):
    maze = Maze(size=size, shape=shape, solver=solver)
    return [maze.slits[-1] + 5, maze.arena_height / 2, 0]


class Maze(b2World):
    def __init__(self, *args, size='XL', shape='SPT', solver='ant', free=False, position=None, angle=0,
                 point_particle=False, geometry: tuple = None, i=0):
        super().__init__(gravity=(0, 0), doSleep=True)

        if len(args) > 0 and type(args[0]).__name__ in ['Trajectory_human', 'Trajectory_ps_simulation',
                                                        'Trajectory_ant', 'Trajectory_gillespie', 'Trajectory',
                                                        'Trajectory_part']:
            x = args[0]
            self.shape = x.shape  # loadshape (maybe this will become name of the maze...)
            self.size = x.size  # size
            self.solver = x.solver
            if position is None:
                position = x.position[i]
            if angle is None:
                angle = x.angle[i]
            self.excel_file_maze, self.excel_file_load = x.geometry()

        else:
            is_exp_valid(shape, solver, size)
            self.shape = shape  # load shape (maybe this will become name of the maze...)
            self.solver = solver  # load shape (maybe this will become name of the maze...)
            self.size = size
            if geometry is None:
                raise ValueError('You have to pass a geometry')
            self.excel_file_maze, self.excel_file_load = geometry

        self.free = free
        self.arena_length = float()
        self.arena_height = float()
        self.exit_size = float()
        self.wallthick = float()
        self.slits = list()
        self.slitpoints = np.array([])
        self.slitTree = list()
        self.body = self.create_Maze()
        self.get_zone()

        self.create_Load(position=position, angle=angle, point_particle=point_particle)

    def getMazeDim(self):
        if self.free:
            self.arena_height = 10
            self.arena_length = 10
            return

        else:
            df = read_excel(path.join(maze_dimension_directory, self.excel_file_maze), engine='openpyxl')

            if self.solver in ['ant', 'ps_simulation', 'sim', 'gillespie']:  # all measurements in cm
                d = df.loc[df['Name'] == self.size + '_' + self.shape]
                self.arena_length = d['arena_length'].values[0]
                self.arena_height = d['arena_height'].values[0]
                self.exit_size = d['exit_size'].values[0]
                self.wallthick = d['wallthick'].values[0]
                if type(d['slits'].values[0]) == str:
                    self.slits = [[float(s) for s in d['slits'].values[0].split(', ')][0],
                                  [float(s) for s in d['slits'].values[0].split(', ')][1]]
                else:
                    self.slits = [d['slits'].values[0]]

            elif self.solver == 'humanhand':  # only SPT
                d = df.loc[df['Name'] == self.solver]
                self.arena_length = d['arena_length'].values[0]
                self.arena_height = d['arena_height'].values[0]
                self.exit_size = d['exit_size'].values[0]
                self.wallthick = d['wallthick'].values[0]
                self.slits = [float(s) for s in d['slits'].values[0].split(', ')]

            elif self.solver == 'human':  # all measurements in meters
                # StartedScripts: measure the slits again...
                # these coordinate values are given inspired from the drawing in \\phys-guru-cs\ants\Tabea\Human
                # Experiments\ExperimentalSetup
                d = df.loc[df['Name'] == self.size]
                A = [float(s) for s in d['A'].values[0].split(',')]
                # B = [float(s) for s in d['B'].values[0].split(',')]
                C = [float(s) for s in d['C'].values[0].split(',')]
                D = [float(s) for s in d['D'].values[0].split(',')]
                E = [float(s) for s in d['E'].values[0].split(',')]

                self.arena_length, self.exit_size = A[0], D[1] - C[1]
                self.wallthick = 0.1
                self.arena_height = 2 * C[1] + self.exit_size
                self.slits = [(E[0] + self.wallthick / 2),
                              (C[0] + self.wallthick / 2)]  # These are the x positions at which the slits are positions

            self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)

    def create_Maze(self):
        self.getMazeDim()
        my_maze = self.CreateBody(b2BodyDef(position=(0, 0), angle=0, type=b2_staticBody, userData='maze'))

        if self.free:
            my_maze.CreateLoopFixture(
                vertices=[(0, 0), (0, self.arena_height * 3), (self.arena_length * 3, self.arena_height * 3),
                          (self.arena_length * 3, 0)])
        else:
            my_maze.CreateLoopFixture(
                vertices=[(0, 0), (0, self.arena_height), (self.arena_length, self.arena_height),
                          (self.arena_length, 0)])
            self.CreateSlitObject(my_maze)
        return my_maze

    def corners(self):
        corners = [[0, 0],
                   [0, self.arena_height],
                   [self.slits[-1] + 20, self.arena_height],
                   [self.slits[-1] + 20, 0],
                   ]
        return np.array(corners + list(np.resize(self.slitpoints, (16, 2))))

    def CreateSlitObject(self, my_maze):
        # # The x and y position describe the point, where the middle (in x direction) of the top edge (y direction)
        # of the lower wall of the slit is...
        if self.shape == 'LongT':
            # TODO
            pass

        # We need a special case for L_SPT because in the manufacturing the slits were not vertically glued
        if self.size == 'L' and self.shape == 'SPT' and self.excel_file_maze == 'MazeDimensions_ant_old.xlsx':
            slitLength = 4.1
            # this is the left (inside), bottom Slit
            self.slitpoints[0] = np.array([[self.slits[0], 0],
                                           [self.slits[0], slitLength],
                                           [self.slits[0] + self.wallthick, slitLength],
                                           [self.slits[0] + self.wallthick, 0]]
                                          )
            # this is the left (inside), upper Slit
            self.slitpoints[1] = np.array([[self.slits[0] - 0.05, slitLength + self.exit_size],
                                           [self.slits[0] + 0.1, self.arena_height],
                                           [self.slits[0] + self.wallthick + 0.1, self.arena_height],
                                           [self.slits[0] + self.wallthick - 0.05, slitLength + self.exit_size]]
                                          )

            # this is the right (outside), lower Slit
            self.slitpoints[2] = np.array([[self.slits[1], 0],
                                           [self.slits[1] + 0.1, slitLength],
                                           [self.slits[1] + self.wallthick + 0.1, slitLength],
                                           [self.slits[1] + self.wallthick, 0]]
                                          )
            # this is the right (outside), upper Slit
            self.slitpoints[3] = np.array([[self.slits[1] + 0.2, slitLength + self.exit_size],
                                           [self.slits[1] + 0.2, self.arena_height],
                                           [self.slits[1] + self.wallthick + 0.2, self.arena_height],
                                           [self.slits[1] + self.wallthick + 0.2, slitLength + self.exit_size]]
                                          )

        # I am not sure, that I need this.
        # # elif size == 'M' or size == 'XL'
        # elif self.shape == 'SPT':
        #     slitLength = (self.arena_height - self.exit_size) / 2
        #     # this is the left (inside), bottom Slit
        #     self.slitpoints[0] = np.array([[self.slits[0], 0],
        #                                    [self.slits[0], slitLength],
        #                                    [self.slits[0] + self.wallthick, slitLength],
        #                                    [self.slits[0] + self.wallthick, 0]]
        #                                   )
        #     # this is the left (inside), upper Slit
        #     self.slitpoints[1] = np.array([[self.slits[0], slitLength + self.exit_size],
        #                                    [self.slits[0], self.arena_height],
        #                                    [self.slits[0] + self.wallthick, self.arena_height],
        #                                    [self.slits[0] + self.wallthick, slitLength + self.exit_size]]
        #                                   )
        #
        #     # this is the right (outside), lower Slit
        #     self.slitpoints[2] = np.array([[self.slits[1], 0],
        #                                    [self.slits[1], slitLength],
        #                                    [self.slits[1] + self.wallthick, slitLength],
        #                                    [self.slits[1] + self.wallthick, 0]]
        #                                   )
        #     # this is the right (outside), upper Slit
        #     self.slitpoints[3] = np.array([[self.slits[1], slitLength + self.exit_size],
        #                                    [self.slits[1], self.arena_height],
        #                                    [self.slits[1] + self.wallthick, self.arena_height],
        #                                    [self.slits[1] + self.wallthick, slitLength + self.exit_size]]
        #                                   )
        #
        #     # slit_up
        #     my_maze.CreatePolygonFixture(vertices=self.slitpoints[0].tolist())
        #     my_maze.CreatePolygonFixture(vertices=self.slitpoints[2].tolist())
        #
        #     # slit_down
        #     my_maze.CreatePolygonFixture(vertices=self.slitpoints[1].tolist())
        #     my_maze.CreatePolygonFixture(vertices=self.slitpoints[3].tolist())

        else:
            self.slitpoints = np.empty((len(self.slits) * 2, 4, 2), float)
            for i, slit in enumerate(self.slits):
                # this is the lower Slit
                self.slitpoints[2 * i] = np.array([[slit, 0],
                                                   [slit, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, (self.arena_height - self.exit_size) / 2],
                                                   [slit + self.wallthick, 0]]
                                                  )

                my_maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i].tolist())

                # this is the upper Slit
                self.slitpoints[2 * i + 1] = np.array([[slit, (self.arena_height + self.exit_size) / 2],
                                                       [slit, self.arena_height],
                                                       [slit + self.wallthick, self.arena_height],
                                                       [slit + self.wallthick,
                                                        (self.arena_height + self.exit_size) / 2]]
                                                      )

                my_maze.CreatePolygonFixture(vertices=self.slitpoints[2 * i + 1].tolist())

            # I dont want to have the vertical line at the first exit
            self.slitTree = BoxIt(np.array([[0, 0],
                                            [0, self.arena_height],
                                            [self.slits[-1], self.arena_height],
                                            [self.slits[-1], 0]]),
                                  0.1, without='right')

            for slit_points in self.slitpoints:
                self.slitTree = np.vstack((self.slitTree, BoxIt(slit_points, 0.01)))

            self.slitTree = cKDTree(self.slitTree)

    def get_zone(self):
        if self.free:
            self.zone = np.empty([0, 2])
            return
        if self.shape == 'SPT':
            self.zone = np.array([[0, 0],
                                  [0, self.arena_height],
                                  [self.slits[0], self.arena_height],
                                  [self.slits[0], 0]])
        else:
            RF = ResizeFactors[self.solver][self.size]
            self.zone = np.array(
                [[self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 - self.arena_height * RF / 2],
                 [self.slits[0] - self.arena_length * RF / 2, self.arena_height / 2 + self.arena_height * RF / 2],
                 [self.slits[0], self.arena_height / 2 + self.arena_height * RF / 2],
                 [self.slits[0], self.arena_height / 2 - self.arena_height * RF / 2]])
        return

    # def possible_state_transitions(self, From, To):
    #     transitions = dict()
    #
    #     s = self.statenames
    #     if self.shape == 'H':
    #         transitions[s[0]] = [s[0], s[1], s[2]]
    #         transitions[s[1]] = [s[1], s[0], s[2], s[3]]
    #         transitions[s[2]] = [s[2], s[0], s[1], s[4]]
    #         transitions[s[3]] = [s[3], s[1], s[4], s[5]]
    #         transitions[s[4]] = [s[4], s[2], s[3], s[5]]
    #         transitions[s[5]] = [s[5], s[3], s[4]]
    #         return transitions[self.states[-1]].count(To) > 0
    #
    #     if self.shape == 'SPT':
    #         transitions[s[0]] = [s[0], s[1]]
    #         transitions[s[1]] = [s[1], s[0], s[2]]
    #         transitions[s[2]] = [s[2], s[1], s[3]]
    #         transitions[s[3]] = [s[3], s[2], s[4]]
    #         transitions[s[4]] = [s[4], s[3], s[5]]
    #         transitions[s[5]] = [s[5], s[4], s[6]]
    #         transitions[s[6]] = [s[6], s[5]]
    #         return transitions[self.states[From]].count(To) > 0

    def set_configuration(self, position, angle):
        self.bodies[-1].position.x, self.bodies[-1].position.y, self.bodies[-1].angle = position[0], position[1], angle

    def minimal_path_length(self):
        from DataFrame.dataFrame import myDataFrame
        from trajectory_inheritance.trajectory_ps_simulation import filename_dstar
        p = myDataFrame.loc[myDataFrame['filename'] == filename_dstar(self.size, self.shape, 0, 0)][['path length [length unit]']]
        return p.values[0][0]

    def create_Load(self, position=None, angle=0, point_particle=False):
        if position is None:
            position = [0, 0]
        self.CreateBody(b2BodyDef(position=(float(position[0]), float(position[1])),
                                  angle=float(angle),
                                  type=b2_dynamicBody,
                                  fixedRotation=False,
                                  linearDamping=0,
                                  angularDamping=0,
                                  userData='load'),
                        restitution=0,
                        friction=0,
                        )

        self.addLoadFixtures(point_particle=point_particle)

    def addLoadFixtures(self, point_particle=False):
        if point_particle:
            return

        my_load = self.bodies[-1]
        if self.shape == 'circle':
            from trajectory_inheritance.gillespie import radius
            my_load.CreateFixture(b2FixtureDef(shape=b2CircleShape(pos=(0, 0), radius=radius)),
                                  density=1, friction=0, restitution=0,
                                  )

        if self.shape == 'H':
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2),
                (shape_width / 2, -shape_thickness / 2),
                (-shape_width / 2, -shape_thickness / 2),
                (-shape_width / 2, shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2),
                (shape_width / 2, shape_height / 2),
                (shape_width / 2 - shape_thickness, shape_height / 2),
                (shape_width / 2 - shape_thickness, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2),
                (-shape_width / 2, shape_height / 2),
                (-shape_width / 2 + shape_thickness, shape_height / 2),
                (-shape_width / 2 + shape_thickness, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'I':
            [shape_height, _, shape_thickness] = self.getLoadDim()
            my_load.CreatePolygonFixture(vertices=[
                (shape_height / 2, -shape_thickness / 2),
                (shape_height / 2, shape_thickness / 2),
                (-shape_height / 2, shape_thickness / 2),
                (-shape_height / 2, -shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'T':
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            resize_factor = ResizeFactors[self.solver][self.size]
            h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower force_vector of the T.

            #  Top horizontal T force_vector
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, -shape_width / 2),
                ((-shape_height - shape_thickness) / 2 + h, shape_width / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_width / 2)],
                density=1, friction=0, restitution=0,
            )

            #  Bottom vertical T force_vector
            my_load.CreatePolygonFixture(vertices=[
                ((-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, -shape_thickness / 2),
                ((shape_height - shape_thickness) / 2 + h, shape_thickness / 2),
                ((-shape_height + shape_thickness) / 2 + h, shape_thickness / 2)], density=1, friction=0, restitution=0)

        if self.shape == 'SPT':  # This is the Special T
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim()

            # h = SPT_centroid_shift * ResizeFactors[x.size]  # distance of the centroid away from the center of the
            # long middle
            h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
            # force_vector of the T. (1.445 calculated)

            # This is the connecting middle piece
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2 - h, shape_thickness / 2),
                (shape_width / 2 - h, -shape_thickness / 2),
                (-shape_width / 2 - h, -shape_thickness / 2),
                (-shape_width / 2 - h, shape_thickness / 2)],
                density=1, friction=0, restitution=0,
            )

            # This is the short side
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2 - h, -short_edge / 2),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2 - h, short_edge / 2),
                (shape_width / 2 - shape_thickness - h, short_edge / 2),
                (shape_width / 2 - shape_thickness - h, -short_edge / 2)],
                density=1, friction=0, restitution=0,
            )

            # This is the long side
            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2 - h, -shape_height / 2),
                (-shape_width / 2 - h, shape_height / 2),
                (-shape_width / 2 + shape_thickness - h, shape_height / 2),
                (-shape_width / 2 + shape_thickness - h, -shape_height / 2)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'RASH':  # This is the ASymmetrical H
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            assymetric_h_shift = ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][self.size]
            # I multiply all these values with 2, because I got them in L, but want to state
            # them in XL.
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2,),
                (shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, shape_thickness / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2, shape_height / 2,),
                (shape_width / 2 - shape_thickness, shape_height / 2,),
                (shape_width / 2 - shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2,),
                (-shape_width / 2, shape_height / 2 - assymetric_h_shift,),
                (-shape_width / 2 + shape_thickness, shape_height / 2 - assymetric_h_shift,),
                (-shape_width / 2 + shape_thickness, -shape_height / 2,)],
                density=1, friction=0, restitution=0,
            )

        if self.shape == 'LASH':  # This is the ASymmetrical H
            [shape_height, shape_width, shape_thickness] = self.getLoadDim()
            assymetric_h_shift = ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][self.size]
            # I multiply all these values with 2, because I got them in L, but want to state
            # them in XL.
            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, shape_thickness / 2,),
                (shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, -shape_thickness / 2,),
                (-shape_width / 2, shape_thickness / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (shape_width / 2, -shape_height / 2,),
                # This addition is because the special T looks like an H where one vertical side is shorter by a factor
                # SPT_ratio
                (shape_width / 2, shape_height / 2 - assymetric_h_shift,),
                (shape_width / 2 - shape_thickness, shape_height / 2 - assymetric_h_shift,),
                (shape_width / 2 - shape_thickness, -shape_height / 2,)],
                density=1, friction=0, restitution=0,
            )

            my_load.CreatePolygonFixture(vertices=[
                (-shape_width / 2, -shape_height / 2 + assymetric_h_shift,),
                (-shape_width / 2, shape_height / 2,),
                (-shape_width / 2 + shape_thickness, shape_height / 2,),
                (-shape_width / 2 + shape_thickness, -shape_height / 2 + assymetric_h_shift,)],
                density=1, friction=0, restitution=0,
            )
        return my_load

    def getLoadDim(self):
        df = read_excel(path.join(maze_dimension_directory, self.excel_file_load), engine='openpyxl')

        if self.shape != 'SPT' and self.solver in ant_dimensions:
            d = df.loc[df['Name'] == self.shape]
            shape_sizes = [d['height'].values[0], d['width'].values[0], d['thickness'].values[0]]
            resize_factor = ResizeFactors[self.solver][self.size]
            dimensions = [i * resize_factor for i in shape_sizes]

            if (resize_factor == 1) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [8.14, 5.6, 1.2]]
            elif (resize_factor == 0.75) and self.shape[1:] == 'ASH':  # for XL ASH
                dimensions = [le * resize_factor for le in [9, 6.2, 1.2]]
            return dimensions

        if self.solver in ant_dimensions:
            d = df.loc[df['Name'] == self.size + '_' + self.shape]

        elif self.solver == 'human':
            d = df.loc[df['Name'] == self.size[0]]

        elif self.solver == 'humanhand':
            d = df.loc[0]
        else:
            raise ValueError('Unclear Load dimensions')

        dimensions = [d['long edge'].values[0], d['length'].values[0], d['width'].values[0],
                      d['short edge'].values[0]]
        return dimensions

    def force_attachment_positions_in_trajectory(self, x, reference_frame='maze'):
        """
        force attachment in world coordinates
        """
        initial_pos, initial_angle = copy(self.bodies[-1].position), copy(self.bodies[-1].angle)
        if reference_frame == 'maze':
            force_attachment_positions_in_trajectory = []
            for i in range(len(x.frames)):
                self.set_configuration(x.position[i], x.angle[i])
                force_attachment_positions_in_trajectory.append(self.force_attachment_positions())
            self.set_configuration(initial_pos, initial_angle)
            return np.array(force_attachment_positions_in_trajectory)
        elif reference_frame == 'load':
            self.set_configuration([0, 0], 0)
            force_attachment = np.stack([self.force_attachment_positions() for _ in range(len(x.frames))])
            self.set_configuration(initial_pos, initial_angle)
            return np.array(force_attachment)
        else:
            raise ValueError('Unknown reference frame!')

    def force_attachment_positions(self):
        from trajectory_inheritance.humans import participant_number
        if self.solver == 'human' and self.size == 'Medium' and self.shape == 'SPT':
            # Aviram went counter clockwise in his analysis. I fix this using Medium_id_correction_dict
            [shape_height, shape_width, shape_thickness, _] = self.getLoadDim()
            x29, x38, x47 = (shape_width - 2 * shape_thickness) / 4, 0, -(shape_width - 2 * shape_thickness) / 4

            # (0, 0) is the middle of the shape
            positions = [[shape_width / 2, 0],
                         [x29, shape_thickness / 2],
                         [x38, shape_thickness / 2],
                         [x47, shape_thickness / 2],
                         [-shape_width / 2, shape_height / 4],
                         [-shape_width / 2, -shape_height / 4],
                         [x47, -shape_thickness / 2],
                         [x38, -shape_thickness / 2],
                         [x29, -shape_thickness / 2]]
            h = centerOfMass_shift * shape_width

        elif self.solver == 'human' and self.size == 'Large' and self.shape == 'SPT':
            [shape_height, shape_width, shape_thickness, short_edge] = self.getLoadDim()

            xMNOP = -shape_width / 2
            xLQ = xMNOP + shape_thickness / 2
            xAB = (-1) * xMNOP
            xCZ = (-1) * xLQ
            xKR = xMNOP + shape_thickness
            xJS, xIT, xHU, xGV, xFW, xEX, xDY = [xKR + (shape_width - 2 * shape_thickness) / 8 * i for i in range(1, 8)]

            yA_B = short_edge / 6
            yC_Z = short_edge / 2
            yDEFGHIJ_STUVWXY = shape_thickness / 2
            yK_R = shape_height / 10 * 3  # TODO: Tabea, you changed this
            yL_Q = shape_height / 2
            yM_P = shape_height / 10 * 3
            yN_O = shape_height / 10

            # indices_to_coords in comment describe the index shown in Aviram's tracking movie
            positions = [[xAB, -yA_B],  # 1, A
                         [xAB, yA_B],  # 2, B
                         [xCZ, yC_Z],  # 3, C
                         [xDY, yDEFGHIJ_STUVWXY],  # 4, D
                         [xEX, yDEFGHIJ_STUVWXY],  # 5, E
                         [xFW, yDEFGHIJ_STUVWXY],  # 6, F
                         [xGV, yDEFGHIJ_STUVWXY],  # 7, G
                         [xHU, yDEFGHIJ_STUVWXY],  # 8, H
                         [xIT, yDEFGHIJ_STUVWXY],  # 9, I
                         [xJS, yDEFGHIJ_STUVWXY],  # 10, J
                         [xKR, yK_R],  # 11, K
                         [xLQ, yL_Q],  # 12, L
                         [xMNOP, yM_P],  # 13, M
                         [xMNOP, yN_O],  # 14, N
                         [xMNOP, -yN_O],  # 15, O
                         [xMNOP, -yM_P],  # 16, P
                         [xLQ, -yL_Q],  # 17, Q
                         [xKR, -yK_R],  # 18, R
                         [xJS, -yDEFGHIJ_STUVWXY],  # 19, S
                         [xIT, -yDEFGHIJ_STUVWXY],  # 20, T
                         [xHU, -yDEFGHIJ_STUVWXY],  # 21, U
                         [xGV, -yDEFGHIJ_STUVWXY],  # 22, V
                         [xFW, -yDEFGHIJ_STUVWXY],  # 23, W
                         [xEX, -yDEFGHIJ_STUVWXY],  # 24, X
                         [xDY, -yDEFGHIJ_STUVWXY],  # 25, Y
                         [xCZ, -yC_Z],  # 26, Z
                         ]
            h = centerOfMass_shift * shape_width

        else:
            positions = [[0, 0] for i in range(participant_number[self.size])]
            h = 0

        # centerOfMass_shift the shape...
        positions = [[r[0] - h, r[1]] for r in positions]  # r vectors in the load frame
        return np.array(
            [np.array(self.bodies[-1].GetWorldPoint(b2Vec2(r))) for r in positions])  # r vectors in the lab frame

    def draw(self, display=None):
        if display is None:
            from PhysicsEngine.Display import Display
            d = Display('', self)
        else:
            d = display
        for body in self.bodies:
            for fixture in body.fixtures:
                if str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2PolygonShape'>":
                    vertices = [(body.transform * v) for v in fixture.shape.vertices]
                    Polygon(vertices, color=colors[body.userData]).draw(d)

                elif str(type(fixture.shape)) == "<class 'Box2D.Box2D.b2CircleShape'>":
                    position = body.position + fixture.shape.pos
                    Circle(position, fixture.shape.radius, color=colors[body.userData]).draw(d)

            if body.userData == 'load':
                Point(np.array(body.position)).draw(d)
        if display is None:
            d.display()

    def average_radius(self):
        r = ResizeFactors[self.solver][self.size]
        radii = {'H': 2.9939 * r,
                 'I': 2.3292 * r,
                 'T': 2.9547 * r,
                 'SPT': 0.76791 * self.getLoadDim()[1],
                 'RASH': 2 * 1.6671 * r,
                 'LASH': 2 * 1.6671 * r}
        return radii[self.shape]

    def circumference(self):
        if self.shape == 'SPT':
            shape_height, shape_width, shape_thickness, shape_height_short_edge = self.getLoadDim()
        else:
            shape_height, shape_width, shape_thickness = self.getLoadDim()

        if self.shape.endswith('ASH'):
            print('I dont know circumference of ASH!!!')
            breakpoint()
        cir = {'H': 4 * shape_height - 2 * shape_thickness + 2 * shape_width,
               'I': 2 * shape_height + 2 * shape_width,
               'T': 2 * shape_height + 2 * shape_width,
               'SPT': 2 * shape_height_short_edge +
                      2 * shape_height -
                      2 * shape_thickness +
                      2 * shape_width,
               'RASH': 2 * shape_width + 4 * shape_height - 4 * ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][
                   self.size]
                       - 2 * shape_thickness,
               'LASH': 2 * shape_width + 4 * shape_height - 4 * ASSYMETRIC_H_SHIFT * ResizeFactors[self.solver][
                   self.size]
                       - 2 * shape_thickness
               }
        return cir[self.shape]
