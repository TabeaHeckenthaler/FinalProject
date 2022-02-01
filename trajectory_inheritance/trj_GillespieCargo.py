from trajectory_inheritance.trajectory import Trajectory
import numpy as np
from trajectory_inheritance.GillespieCargo import GillespieCargo
from Setup.Maze import Maze
from PhysicsEngine.drawables import Arrow, Line
# from Box2D import b2Body

time_step = 0.01


class Trajectory_gillespie(Trajectory):
    def __init__(self, size=None, shape=None, filename='gillespie_test', fps=time_step, winner=bool, free=False):

        solver = 'gillespie'
        super().__init__(size=size, shape=shape, solver=solver, filename=filename, fps=fps, winner=winner)
        self.free = free

        maze = Maze(size=size, shape=shape, solver=solver, free=free)
        # slit middle coordinates for informed ants
#        x_info, y_info = maze.slits[0], maze.arena_height / 2
        self.GillespieCargo = GillespieCargo(maze)

    def step(self, maze, i, display=None):

        maze.set_configuration(self.position[i], self.angle[i])
        cargo = maze.bodies[-1]

        if self.GillespieCargo.dt_event < time_step:
            self.GillespieCargo.dt_event = self.GillespieCargo.next_event(cargo)

        self.forces(cargo, display=display)

        self.GillespieCargo.dt_event -= time_step
        maze.Step(time_step, 10, 10)

        self.position = np.vstack((self.position, [cargo.position.x, cargo.position.y]))
        self.angle = np.hstack((self.angle, cargo.angle))
        return

    def forces(self, cargo, display=None):
        # TODO: make this better...
        cargo.linearVelocity = 0 * cargo.linearVelocity
        cargo.angularVelocity = 0 * cargo.angularVelocity

        """ Magnitude of forces """
        for i in range(len(self.GillespieCargo.n_p)):
            start = self.GillespieCargo.pos_site(i, cargo)

            if self.GillespieCargo.n_p[i]:
                f_ant = self.GillespieCargo.ant_force(i, cargo)
                cargo.ApplyForce(f_ant, start, True)
                if display is not None:
                    Arrow(start, start + [0.5 * f_ant[0], 0.5 * f_ant[1]], 'puller', 'puller').draw(display)

            elif self.GillespieCargo.n_l[i]:
                vec = self.GillespieCargo.normal_site_vec(i, cargo.angle)  # TODO check this vector
                if display is not None:
                    Line(start, start + [0.5 * vec[0], 0.5 * vec[1]]).draw(display)

#            else:
#                Point(start, end).draw(display)

    def run_simulation(self, frameNumber, free=False, display=True):
        maze = Maze(size=self.size, shape=self.shape, solver='sim', free=free)
        self.frames = np.linspace(1, frameNumber, frameNumber)
        self.position = np.array([[maze.arena_length / 4, maze.arena_height / 2]])
        self.angle = np.array([0], dtype=float)  # array to store the position and angle of the load
        if display:
            from PhysicsEngine.Display import Display
            self.run_trj(maze, display=Display(self.filename, maze))
        else:
            self.run_trj(maze)

    def load_participants(self):
        self.participants = self.GillespieCargo

    def averageCarrierNumber(self):
        return self.GillespieCargo.N_max  # TODO: this is maximum, not average...

