# similar to the Gillespie Code described in the SI of the nature communications paper
# titled 'Ants optimally amplify... '
import numpy as np
from Box2D import b2Vec2
from Analysis.GeneralFunctions import rot
from Setup.Load import init_sites

"""
Parameters I chose
"""

# ant force [N]
f_0 = 1

"""
Parameters from the paper
"""
# detachment rate from a moving cargo, detachment from a non moving cargo [1/s]
k1_off, k2_off = 0.035, 0.01
# attachment independent of cargo velocity [1/s]
k_on = 0.0017
# Radius of the object
radius = 0.57
# Number of available sites
N_max = 20
#
k_c = 0.7
# beta, I am not sure, what this is
beta = 1.65
# connected to the ants decision making process
f_ind = 10 * f_0
# kinetic friction linear coefficient [N * sec/cm]
gamma = 25 * f_0
# kinetic rotational friction coefficient [N * sec/c]
gamma_rot = 0.4 * gamma

f_kinx, f_kiny = 0, 0
n_av = 100  # number of available ants
phi_max = 0.9075712110370514  # rad in either direction from the normal


class Gillespie:
    def __init__(self, my_maze):
        self.n_p = [0 for _ in range(N_max)]  # array with 0 and 1 depending on whether is puller or not
        self.n_l = [0 for _ in range(N_max)]  # array with 0 and 1 depending on whether is lifter of not

        self._attachment_sites, self._phi_default_load_coord = init_sites(my_maze, N_max)
        # vector to ith attachment site in load coordinates
        # angle of normal vector from ith attachment site to the x axis of the world, when load.angle = 0

        self.phi = np.empty(N_max)  # angle of the ant to world coordinates!
        self.phi[:] = np.nan  # (NaN if not occupied)

        self.r_att = None
        self.r_det = None
        self.r_con = None
        self.r_orient = None
        self.r_tot = None

        self.time_until_next_event = 0
        self.populate(my_maze.bodies[-1])

    @property
    def attachment_sites(self):
        return self._attachment_sites

    @attachment_sites.setter
    def attachment_sites(self, value):
        raise AttributeError('Dont change attachment sites!')

    @property
    def phi_default_load_coord(self):
        return self._phi_default_load_coord

    @phi_default_load_coord.setter
    def phi_default_load_coord(self, value):
        raise AttributeError('Dont change default phis!')

    def is_occupied(self, *i):

        if len(i) > 0:
            return self.n_p[i[0]] or self.n_l[i[0]]
        else:
            return np.any([self.n_p, self.n_l], axis=0)

    def f_loc(self, my_load, i: int):
        """
        :param my_load: b2Body, which is carried by ants
        :param i: ith attachment position
        :return: Linear velocity of load at the attachment site (b2Vec) in world coordinates so that they will oppose
        rotations
        """
        f_x, f_y = gamma * np.array(my_load.linearVelocity) \
                   + 0.7 * \
                   np.cross(np.hstack([self.attachment_site_world_coord(my_load, i), [0]]),
                            np.array([0, 0, my_load.angularVelocity]))[:2]
        # load.GetLinearVelocityFromLocalPoint(self.attachment_position(load, i))
        # load.ApplyForce
        return f_x, f_y

    def attachment(self, i: int, my_load, ant_type: str, normal: bool = False):
        """
        :param i: site index on which to attach
        :param my_load: b2Body
        :param normal: ant attaches normal to the load
        :return:
        """
        if ant_type == 'puller':
            self.n_p[i] = 1
            f_x, f_y = self.f_loc(my_load, i)
            if normal:
                self.phi[i] = self.phi_default_load_coord[i] + my_load.angle
            else:
                self.phi[i] = np.arctan2(f_y, f_x)

            # When a puller ant is attached to the cargo she contributes
            # to the cargo’s velocity by applying a force,
            # and gets aligned as much as possible with the
            # direction of the local force at its point of attachment.

        else:
            self.n_l[i] = 1
            self.phi[i] = self.phi_default_load_coord[i] + my_load.angle
            # If a lifter ant attaches, she aligns with the outgoing normal of her attachment site

    def detachment(self, i: int):
        """
        :param i: detachment of ant at site i
        :return:
        """
        if not self.is_occupied(i):
            raise ValueError('Detachment at empty site')
        self.n_p[i] = 0
        self.n_l[i] = 0
        self.phi[i] = np.NaN

    def number_attached(self):
        """
        :return: the number of attached ants
        """
        return np.sum(self.is_occupied())

    def number_empty(self):
        """
        :return: number of empty sites
        """
        return N_max - np.sum(self.is_occupied())

    def attachment_site_world_coord(self, my_load, i: int):
        """
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: the ith attachment site on the object
        :return: the attachment position in world coordinates.
        """
        return my_load.position + np.dot(rot(my_load.angle), self.attachment_sites[i])

    def normal_site_vector(self, angle: float, i: int):
        """
        :param i: ith position, counted clockwise
        :param angle: angle of the shape to the world coordinate system (load.angle)
        :return: ant vector pointing in the direction that the ant is pointing in world coordinate system
        """
        vector = np.array([np.cos(self.phi_default_load_coord[i]), np.sin(self.phi_default_load_coord[i])])
        return np.dot(rot(angle), vector)

    def ant_vector(self, angle: float, i: int):
        """
        :param i: ith position, counted clockwise
        :param angle: angle of the shape to the world coordinate system (load.angle)
        :return: ant vector pointing in the direction that the ant is pointing in world coordinates
        """
        vector = np.array([np.cos(self.phi[i]), np.sin(self.phi[i])])
        return np.dot(rot(angle), vector)

    def ant_force(self, my_load, i: int, pause=False):
        """
        Updates my_loads linear and angular velocity and returns the force vector in world coordinate system
        :param my_load: b2Body, which is the object moved by the ants. It contains fixtures,
         which indicate the extent of the body.
        :param i: ith position, counted counter clockwise
        :param pause: boolean. Whether the force should be applied, or we just return current force vector
        :return: force vector (np.array) pointing along the body axis of the ant at the ith position
        """
        if not self.n_p[i]:
            raise ValueError('Force originating from a non puller site')

        force = np.array(f_0 / gamma * self.ant_vector(my_load.angle, i))

        # equations (4) and (5) from the SI
        if not pause:
            vectors = (self.attachment_site_world_coord(my_load, i) - my_load.position,
                       self.ant_vector(my_load.angle, i))
            my_load.linearVelocity = my_load.linearVelocity + f_0 / gamma * b2Vec2(np.inner(*vectors), 0)
            my_load.angularVelocity = my_load.angularVelocity + f_0 / gamma_rot * np.cross(*vectors)
        # TODO: this is not correct
        return force

    def whatsNext(self, my_load):
        """
        Decides the new change in the ant configuration. Randomly according to calculated probabilities one of the
        following events occur:
        (1) attachment of a new ant
        (2) detachment of an attached ant
        (3) conversion of a puller to lifter, or lifter to puller
        (4) reorientation of an ant to pull in the current direction of motion
        :param my_load: b2Body
        :return:
        """
        lot = np.random.uniform(0, 1)
        self.update_rates(my_load)

        # new attachment
        if lot < self.r_att / self.r_tot:
            i = np.random.choice(np.where([not occ for occ in self.is_occupied()])[0])
            self.new_attachment(i, my_load)

        # detachment
        elif lot < (self.r_att + self.r_det) / self.r_tot:
            i = np.random.choice(np.where(self.is_occupied())[0])
            self.detachment(i)

        # conversion of lifter to puller or vice versa
        elif lot < (self.r_att + self.r_det + self.r_con) / self.r_tot:
            def rl_p(ii):
                return k_c * np.exp(np.inner(self.normal_site_vector(my_load.angle, ii),
                                             self.f_loc(my_load, ii)) / f_ind)

            def rp_l(ii):
                rp_l = k_c * np.exp(-np.inner(self.normal_site_vector(my_load.angle, ii),
                                              self.f_loc(my_load, ii)) / f_ind)
                if np.isnan(rp_l):
                    k_c * np.exp(-np.inner(self.normal_site_vector(my_load.angle, ii),
                                           self.f_loc(my_load, ii)) / f_ind)
                return rp_l

            prob_unnorm = [self.n_p[ii] * rp_l(ii) + self.n_l[ii] * rl_p(ii) for ii in np.where(self.is_occupied())[0]]
            i = np.random.choice([ii for ii in np.where(self.is_occupied())[0]],
                                 p=prob_unnorm * 1 / np.sum(prob_unnorm))
            if not (self.n_p[i] or self.n_l[i]) or (self.n_p[i] and self.n_l[i]):
                raise ValueError('Switching is messed up!')
            self.n_p[i], self.n_l[i] = self.n_l[i], self.n_p[i]

        # reorientation
        else:
            i = np.random.choice(np.where(self.n_p)[0])
            # TODO: 52 degrees in both directions
            f_x, f_y = self.f_loc(my_load, i)
            if abs((self.phi_default_load_coord[i] + my_load.angle) - self.phi[i]) > phi_max:
                self.phi[i] = np.arctan2(f_y, f_x)
            else:
                print('overstretch!')
        return self.dt()

    def populate(self, my_load):
        """
        populating load with ants
        """
        for i in range(len(self.n_p)):
            self.new_attachment(i, my_load, ant_type='puller', normal=False)

    def new_attachment(self, i: int, my_load, ant_type=None, normal: bool = False):
        """
        A new ant attaches, and becomes either puller or lifter

        :param i: ith attachment position
        :param my_load: Box2D body
        :param ant_type: str, 'puller' or 'lifter' or None
        :param normal: ant attaches normal to the load
        :return:
        """
        if self.is_occupied(i):
            raise ValueError('Ant tried to attach to occupied site')
        if ant_type is not None and ant_type not in ['puller', 'lifter']:
            raise ValueError('unknown type of ant... not puller or lifter')
        if ant_type is None:
            puller = 1 / (1 + np.exp(
                -np.inner(self.normal_site_vector(my_load.angle, i), self.f_loc(my_load, i)) / f_ind))
            if np.random.uniform(0, 1) < puller:
                ant_type = 'puller'
            else:
                ant_type = 'lifter'

        self.attachment(i, my_load, ant_type, normal=normal)
        return

    def update_rates(self, my_load):
        """
        update rates of attachment, detachment, conversion, orientation, and the total (which is the sum of all
        aforementioned)
        :param my_load: Box2D body
        :return:
        """
        self.r_att = k_on * n_av * self.number_empty()
        self.r_det = np.sum([np.sum(self.is_occupied()) * (k1_off * np.heaviside(self.f_loc(my_load, i), 0)
                                                           + k2_off * (1 - np.heaviside(self.f_loc(my_load, i), 0)))
                             for i in range(N_max) if self.is_occupied(i)])
        self.r_con = k_c * np.sum([self.n_p[i] * np.exp(-np.inner(self.normal_site_vector(my_load.angle, i),
                                                                  self.f_loc(my_load, i)) / f_ind)
                                   + self.n_l[i] * np.exp(-np.inner(self.normal_site_vector(my_load.angle, i),
                                                                    self.f_loc(my_load, i)) / f_ind)
                                   for i in np.where(self.is_occupied())[0]])
        self.r_orient = k_c * np.sum(self.n_p)
        self.r_tot = self.r_att + self.r_det + self.r_con + self.r_orient

    def dt(self):
        return -1 / self.r_tot * np.log(np.random.uniform(0, 1))
