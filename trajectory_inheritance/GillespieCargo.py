import math
import numpy as np
from Setup.Load import init_sites

# from Box2D import b2Vec2


class GillespieCargo:
    # model parameters (class variables)
    f0 = 1                          # force applied by a single ant ('ant force')           [Newton]
    N_max = 40                      # number of available attachment sites on the cargo     [sites]
    N_av = 100                      # number of available ('free') ants near cargo          [ants]
    K_on = 0.0017                   # attachment rate to the cargo                          [1/sec]
    K1_off = 0.035                  # detachment rate from a non-moving cargo               [1/sec]
    K2_off = 0.01                   # detachment rate from a moving cargo                   [1/sec]
    K_c = 0.7                       # reorientation and conversion rate between roles       [1/sec]
    F_ind = 10 * f0                 # individuality parameter (effective 'temperature')     [Newton]
    f_s = 2.7 * f0                  # static friction force                                 [Newton]
    f_k = 0.9 * f_s                 # kinetic friction force                                [Newton]
    tau_s = 0.83 * f_s              # static friction torque                                [Newton * cm]
    tau_k = 0.83 * f_k              # kinetic friction torque                               [Newton * cm]
    gamma = 25 * f0                 # cargo linear (force) response coefficient             [Newton * sec / cm]
    gamma_rot = 0.4 * gamma         # cargo rotational (torque) response coefficient        [Newton * sec]
    beta = 1.65                     # factor of friction reduction by a lifter ant          dimensionless
    phi_max = 52 * math.pi / 180    # angular range from both sides of site normal          [rad]

    def __init__(self, cargo):
        # create site occupancy logical arrays
        self.n_p = np.zeros((self.N_max,), dtype=int)  # pullers array ('1' = puller, '0' = not puller)
        self.n_l = np.zeros((self.N_max,), dtype=int)  # lifters array ('1' = lifter, '0' = not lifter)
        # create attachment sites:
        # _pos_sites_def: position vectors of sites in cargo coordinates
        # _angle_sites_def: angles of site normals in cargo coordinates (initial configuration)
        self._pos_sites_def, self._angle_sites_def = init_sites(cargo, self.N_max)
        # initialize site attachment angles
        self.phi = np.empty(self.N_max)
        self.phi[:] = np.NaN  # NaN if not occupied
        # initialize total rates (attachment, detachment, conversion, orientation, total sum)
        self.R_att = self.R_det = self.R_con = self.R_orient = self.R_tot = None
        # initialize time 'dt' to next event
        self.dt_event = 0
        # slit middle coordinates for informed ants
        self.x_info, self.y_info = 13.8, 5

    @property
    def pos_sites_def(self):
        return self._pos_sites_def

    @pos_sites_def.setter
    def pos_sites_def(self, value):
        raise AttributeError("Don't change defined site positions!")

    @property
    def angle_sites_def(self):
        return self._angle_sites_def

    @angle_sites_def.setter
    def angle_sites_def(self, value):
        raise AttributeError("Don't change defined site normal angles!")

    # create a 2x2 rotation matrix
    @staticmethod
    def rot_mat(angle: float):
        """
        Creates a 2x2 rotation matrix to rotate a 2D vector by 'angle' radians.

        :param angle: rotation angle (in radians)
        :return: 2x2 rotation matrix
        """
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    # local force at site 'i'
    # noinspection SpellCheckingInspection
    def f_loc(self, i: int, cargo):
        """
        Calculates the local force at site 'i'.

        :param i: site index (site 'i')
        :param cargo: b2Body, the object moved by the ants
                      (extract linear & angular velocities)
        :return: local force at site 'i' as 2D vector f_loc = [f_x, f_y]
        """
        # linear velocity of the cargo
        # v_cm = cargo.linearVelocity
        # angular velocity of the cargo (as 3D vector, [0, 0, w_z])
        # w = np.array([0, 0, cargo.angularVelocity])
        # position of site 'i' (as 3D vector, [r_x, r_y, 0])
        # r_i = np.hstack([self.pos_site(i, cargo), [0]])
        # local force at site 'i' as 2D vector
        v_loc = cargo.GetLinearVelocityFromWorldPoint(self.pos_site(i, cargo))
        # return self.gamma * (v_cm + 0.7 * np.cross(r_i, w)[:-1])
        return self.gamma * v_loc  # local force is simply proportional to local velocity of the attachment point

    # check which sites are occupied, and which are not
    def is_occupied(self):
        """:return: True if a site is occupied, False otherwise, e.g. [True, False,...]"""
        return np.any([self.n_p, self.n_l], axis=0)

    # indexes of all occupied sites
    def all_attached(self):
        """:return: list of indexes of all occupied sites"""
        return [i for i, x in enumerate(self.is_occupied()) if x]

    # indexes of all empty sites
    def all_empty(self):
        """:return: list of indexes of all empty sites"""
        return [i for i, x in enumerate(self.is_occupied()) if not x]

    # number of occupied sites (attached ants, N_att)
    def num_attached(self):
        """:return: number of attached ants ('N_att')"""
        return np.sum(self.is_occupied())

    # number of empty sites (N_empty)
    def num_empty(self):
        """:return: number of empty sites ('N_empty')"""
        return self.N_max - self.num_attached()

    # attachment of ant at site 'i'
    def attachment(self, i: int, cargo):
        """
        Performs an attachment event at site 'i'.

        :param i: site index (site 'i')
        :param cargo: b2Body, the object moved by the ants
                      (extract angle of cargo)
        """
        # type of newly attached ant (puller or lifter)
        ant_type = self.pull_or_lift(i, cargo)

        if ant_type == 'puller':
            self.n_p[i] = 1
            # local force at site 'i'
            f_loc = self.f_loc(i, cargo)
            # when a puller attaches, she aligns as much as possible with the local force
            self.phi[i] = np.arctan2(f_loc[1], f_loc[0])
            # vec_ant = np.array([self.x_info, self.y_info]) - self.pos_site(i, cargo)
            # self.phi[i] = np.arctan2(vec_ant[1], vec_ant[0])

        elif ant_type == 'lifter':
            self.n_l[i] = 1
            # when a lifter attaches, she aligns with the outgoing site normal
            self.phi[i] = self.angle_sites_def[i] + cargo.angle

        # angle difference between normal direction and orientation
        d_phi = self.angle_sites_def[i] + cargo.angle - self.phi[i]
        # make sure angle difference is in the range [-phi_max, phi_max]
        if d_phi > self.phi_max:
            self.phi[i] = self.angle_sites_def[i] + cargo.angle - self.phi_max
        elif d_phi < - self.phi_max:
            self.phi[i] = self.angle_sites_def[i] + cargo.angle + self.phi_max

    # detachment of ant at site 'i'
    def detachment(self, i: int):
        """
        Performs a detachment event at site 'i'.

        :param i: site index (site 'i')
        """
        self.n_p[i] = 0
        self.n_l[i] = 0
        self.phi[i] = np.NaN

    # conversion of role at site 'i'
    def conversion(self, i: int):
        """
        Performs a conversion event (lifter to puller or puller to lifter) at site 'i'.

        :param i: site index (site 'i')
        """
        self.n_p[i], self.n_l[i] = self.n_l[i], self.n_p[i]

    # reorientation with local force
    def reorient(self, i: int, cargo):
        """
        Performs a reorientation event at site 'i'.

        :param i: site index (site 'i')
        :param cargo: b2Body, the object moved by the ants
                      (extract angle of cargo)
        """
        # local force at site 'i'
        f_loc = self.f_loc(i, cargo)
        # align as much as possible with the local force
        self.phi[i] = np.arctan2(f_loc[1], f_loc[0])
        # angle difference between normal direction and orientation
        d_phi = self.angle_sites_def[i] + cargo.angle - self.phi[i]
        # make sure angle difference is in the range [-phi_max, phi_max]
        if d_phi > self.phi_max:
            self.phi[i] = self.angle_sites_def[i] + cargo.angle - self.phi_max
        elif d_phi < - self.phi_max:
            self.phi[i] = self.angle_sites_def[i] + cargo.angle + self.phi_max

    # type of newly attached ant (puller or lifter)
    def pull_or_lift(self, i: int, cargo):
        """
        Determines if a newly attached ant is a puller or lifter

        :param i: site index (site 'i')
        :param cargo: b2Body, the object moved by the ants
        :return: ant type (puller or lifter)
        """
        # random number from a uniform distribution on (0, 1)
        rand = np.random.uniform(0, 1)
        # probability to become a puller at site 'i'
        prob_pull = 1 / (1 + self.exp_rate(i, cargo))
        # determine ant type (puller or lifter)
        if rand < prob_pull:
            ant_type = 'puller'
        else:
            ant_type = 'lifter'
        return ant_type

    def pos_site(self, i: int, cargo):
        """
        Transforms (by translation and rotation) a site position
        from initial cargo coordinates to absolute 'world' coordinates.

        :param i: site index (site 'i')
        :param cargo: b2Body, the object moved by the ants
                      (extract position & angle of cargo)
        :return: site position in absolute 'world' coordinates
        """
        return cargo.position + np.dot(self.rot_mat(cargo.angle), self.pos_sites_def[i])

    # normal site vector in absolute ('world') coordinates
    def normal_site_vec(self, i: int, angle: float):
        """
        Transforms (by rotation) a normal site vector
        from initial cargo coordinates to absolute 'world' coordinates.

        :param i: site index (site 'i')
        :param angle: angle of cargo to absolute 'world' coordinates
        :return: normal to site 'i' in absolute 'world' coordinates
        """
        vec = np.array([np.cos(self.angle_sites_def[i]), np.sin(self.angle_sites_def[i])])
        # return the vector 'vec' rotated by 'angle' radians
        return np.dot(self.rot_mat(angle), vec)

    # attached ant vector in absolute ('world') coordinates
    def ant_vec(self, i: int, angle: float):
        """
         Transforms (by rotation) an attached ant vector
         from initial cargo coordinates to absolute 'world' coordinates.

         :param i: site index (site 'i')
         :param angle: angle of cargo to absolute 'world' coordinates
         :return: ant vector at site 'i' in absolute 'world' coordinates
         """
        vec = np.array([np.cos(self.phi[i]), np.sin(self.phi[i])])
        # return the vector 'vec' rotated by 'angle' radians
        return vec  # np.dot(self.rot_mat(angle), vec)

    def ant_force(self, i: int, cargo):
        """
        :params i:
        param cargo:
        """
        return np.dot(self.f0, self.ant_vec(i, cargo.angle))

    # determine the next event to take place
    def next_event(self, cargo):
        """
        Determines the next event to take place.

        One of the following events occurs at random,
        according to respective probabilities (rates):
        (1) Attachment of a new ant
        (2) Detachment of an attached ant
        (3) Conversion of a puller to lifter, or a lifter to puller
        (4) Reorientation of a puller in the current direction of motion

        :param cargo: b2Body, the object moved by the ants
        """
        # random number from a uniform distribution on (0, 1)
        rand = np.random.uniform(0, 1)
        # update all rates
        self.update_rates(cargo)

        # (1) Attachment event
        if rand < self.R_att / self.R_tot:
            # choose at random an empty site 'i'
            i = np.random.choice(self.all_empty())
            # apply attachment at site 'i'
            self.attachment(i, cargo)

        # (2) Detachment event
        elif rand < (self.R_att + self.R_det) / self.R_tot:
            # choose at random an occupied site 'i'
            i = np.random.choice(self.all_attached())
            # apply detachment at site 'i'
            self.detachment(i)

        # (3) Conversion event
        elif rand < (self.R_att + self.R_det + self.R_con) / self.R_tot:
            # all individual rates of conversion
            rates = [self.K_c * (self.n_p[i] * self.exp_rate(i, cargo) + self.n_l[i] / self.exp_rate(i, cargo))
                     for i in self.all_attached()]
            # all individual probabilities of conversion
            probs = [x / np.sum(rates) for x in rates]
            # choose at random an occupied site 'i' according to probabilities
            i = np.random.choice(self.all_attached(), 1, p=probs)[0]
            # apply conversion at site 'i'
            self.conversion(i)

        # (4) Reorientation event
        else:
            # choose at random a puller ant
            i = np.random.choice(np.where(self.n_p)[0])
            # apply reorientation at site 'i'
            self.reorient(i, cargo)

        # draw the time to the next event
        return self.next_dt_event()

    # delta = 1 if f_loc > 0, and 0 if f_loc < 0
    def delta(self, i: int, cargo):
        return 1 - np.heaviside(np.linalg.norm(self.f_loc(i, cargo)), 0)

    # exponential factor (used extensively for rates)
    def exp_rate(self, i: int, cargo):
        # normal vector to site 'i'
        p_i = self.normal_site_vec(i, cargo.angle)
        # local force at site 'i'
        f_loc = self.f_loc(i, cargo)
        # exponential factor at site 'i'
        return np.exp(-np.dot(p_i, f_loc) / self.F_ind)

    # update attachment rate
    def update_r_att(self):
        return self.K_on * self.N_av * self.num_empty()

    # update detachment rate
    def update_r_det(self, cargo):
        return np.sum([self.K1_off * self.delta(i, cargo) + self.K2_off * (1 - self.delta(i, cargo))
                       for i in self.all_attached()])

    # update conversion rate
    def update_r_con(self, cargo):
        return self.K_c * np.sum([self.n_p[i] * self.exp_rate(i, cargo) + self.n_l[i] / self.exp_rate(i, cargo)
                                 for i in self.all_attached()])

    # update reorientation rate
    def update_r_orient(self):
        return self.K_c * np.sum(self.n_p)

    def update_rates(self, cargo):
        # attachment rate
        self.R_att = self.update_r_att()
        # detachment rate
        self.R_det = self.update_r_det(cargo)
        # conversion rate
        self.R_con = self.update_r_con(cargo)
        # reorientation rate
        self.R_orient = self.update_r_orient()
        # total rate
        self.R_tot = self.R_att + self.R_det + self.R_con + self.R_orient

    # draw the time to the next event
    def next_dt_event(self):
        # random number from a uniform distribution on (0, 1)
        rand = np.random.uniform(0, 1)
        # time until next event
        return (- 1 / self.R_tot) * np.log(rand)
