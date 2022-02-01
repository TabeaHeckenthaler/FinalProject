from trajectory_inheritance.trajectory import Trajectory
import numpy as np
from trajectory_inheritance.gillespie import Gillespie, N_max
from Setup.Maze import Maze
from PhysicsEngine.drawables import Arrow, Point

time_step = 0.01


# def newFileName(size, shape):
#     counter = int(len(glob.glob(size + '_' + shape + '*_' + '_gillespie_' + '_*')) / 2 + 1)
#     filename = size + '_' + shape + '_gillespie_' + str(counter)
#     return filename


class Trajectory_gillespie(Trajectory):
    def __init__(self, size=None, shape=None, filename='gillespie_test', fps=time_step, winner=bool, free=False):

        solver = 'gillespie'
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.free = free

        my_maze = Maze(size=size, shape=shape, solver=solver, free=free)
        self.gillespie = Gillespie(my_maze)

    def step(self, my_maze, i, display=None):

        my_maze.set_configuration(self.position[i], self.angle[i])

        if self.gillespie.time_until_next_event < time_step:
            self.gillespie.time_until_next_event = self.gillespie.whatsNext(my_maze.bodies[-1])

        self.forces(my_maze.bodies[-1], display=display)

        self.gillespie.time_until_next_event -= time_step
        my_maze.Step(time_step, 10, 10)

        self.position = np.vstack((self.position, [my_maze.bodies[-1].position.x, my_maze.bodies[-1].position.y]))
        self.angle = np.hstack((self.angle, my_maze.bodies[-1].angle))
        return

    def forces(self, my_load, pause=False, display=None):
        # TODO: make this better...
        my_load.linearVelocity = 0 * my_load.linearVelocity
        my_load.angularVelocity = 0 * my_load.angularVelocity

        """ Magnitude of forces """

        for i in range(len(self.gillespie.n_p)):
            start = self.gillespie.attachment_site_world_coord(my_load, i)
            end = None

            if self.gillespie.n_p[i]:
                f_x, f_y = self.gillespie.ant_force(my_load, i, pause=pause)
                Arrow(start, start + [100 * f_x, 100 * f_y], 'puller').draw(display)

            elif self.gillespie.n_l[i]:
                Point(start, end).draw(display)

            else:
                Point(start, end).draw(display)

    def run_simulation(self, frameNumber, free=False):
        my_maze = Maze(size=self.size, shape=self.shape, solver='sim', free=free)
        self.frames = np.linspace(1, frameNumber, frameNumber)
        self.position = np.array([[my_maze.arena_length / 4, my_maze.arena_height / 2]])
        self.angle = np.array([0], dtype=float)  # array to store the position and angle of the load
        from PhysicsEngine.Display import Display
        self.run_trj(my_maze, display=Display(self.filename, my_maze, wait=10))

    def load_participants(self):
        self.participants = self.gillespie

    def averageCarrierNumber(self):
        return N_max  # TODO: this is maximum, not average...

