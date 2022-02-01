import numpy as np
from Box2D import b2CircleShape, b2FixtureDef
from Setup.Maze import ResizeFactors

ant_dimensions = ['ant', 'ps_simulation', 'sim', 'gillespie']  # also in Maze.py
periodicity = {'H': 2, 'I': 2, 'RASH': 2, 'LASH': 2, 'SPT': 1, 'T': 1}

# somehow these contain the same information
SPT_ratio = 2.44 / 4.82


# I multiply all these values with 2, because I got them in L, but want to state them in XL.
# StartedScripts: Does the load have a preferred orientation while moving?


def loops(Box2D_Object, vertices=None):
    if vertices is None:
        vertices = []

    if hasattr(Box2D_Object, 'bodies'):
        for body in Box2D_Object.bodies:
            loops(body, vertices=vertices)
    else:
        for fixture in Box2D_Object.fixtures:  # Here, we update_screen the vertices of our bodies.fixtures and...
            vertices.append(
                [(Box2D_Object.transform * v) for v in fixture.shape.vertices][:4])  # Save vertices of the load
    return vertices


def corners_phis(my_maze):
    """
    Corners are ordered like this: finding the intersection of the negative
    y-axis, and the shape, and going clockwise find the first corner. Then go clockwise in order of the
    corners. corners = np.vstack([[0, corners[0, 1]], corners]) phis = np.append(
    phis, phis[0])

    Phis describe the angles of the normal between the corners to the x axis of the world coordinates.
    Starting at first corner of load.corners, and going clockwise
    load.phis = np.array([np.pi, np.pi / 2, 0, -np.pi / 2])
    """
    if my_maze.shape == 'H':
        [shape_height, shape_width, shape_thickness] = my_maze.getLoadDim()

        corners = np.array([[-shape_width / 2 + shape_thickness, -shape_thickness / 2],
                            [-shape_width / 2 + shape_thickness, -shape_height / 2],
                            [-shape_width / 2, -shape_height / 2],
                            [-shape_width / 2, shape_height / 2],
                            [-shape_width / 2 + shape_thickness, shape_height / 2],
                            [-shape_width / 2 + shape_thickness, shape_thickness / 2],
                            [shape_width / 2 - shape_thickness, shape_thickness / 2],
                            [shape_width / 2 - shape_thickness, shape_height / 2],
                            [shape_width / 2, shape_height / 2],
                            [shape_width / 2, -shape_height / 2],
                            [shape_width / 2 - shape_thickness, -shape_height / 2],
                            [shape_width / 2 - shape_thickness, -shape_thickness / 2]])

        phis = np.array([0, -np.pi / 2, np.pi, np.pi / 2, 0, np.pi / 2,
                         np.pi, np.pi / 2, 0, -np.pi / 2, np.pi, -np.pi / 2])

    if my_maze.shape == 'I':
        [shape_height, _, shape_thickness] = my_maze.getLoadDim()
        corners = np.array([[-shape_height / 2, -shape_thickness / 2],
                            [-shape_height / 2, shape_thickness / 2],
                            [shape_height / 2, shape_thickness / 2],
                            [shape_height / 2, -shape_thickness / 2]])

        phis = np.array([np.pi, np.pi / 2, 0, -np.pi / 2])

    if my_maze.shape == 'T':
        [shape_height, shape_width, shape_thickness] = my_maze.getLoadDim()
        resize_factor = ResizeFactors[my_maze.solver][my_maze.size]
        h = 1.35 * resize_factor  # distance of the centroid away from the center of the lower force_vector of the T.

        corners = np.array([[(shape_height - shape_thickness) / 2 + h, -shape_thickness / 2],
                            [(-shape_height + shape_thickness) / 2 + h, -shape_thickness / 2],
                            [(-shape_height + shape_thickness) / 2 + h, -shape_width / 2],
                            [(-shape_height - shape_thickness) / 2 + h, -shape_width / 2],
                            [(-shape_height - shape_thickness) / 2 + h, shape_width / 2],
                            [(-shape_height + shape_thickness) / 2 + h, shape_width / 2],
                            [(-shape_height + shape_thickness) / 2 + h, shape_thickness / 2],
                            [(shape_height - shape_thickness) / 2 + h, shape_thickness / 2]])

        phis = np.array([np.pi, -np.pi / 2, np.pi, np.pi / 2, 0, -np.pi / 2, 0, -np.pi / 2])

    if my_maze.shape == 'SPT':  # This is the Special T
        [shape_height, shape_width, shape_thickness] = my_maze.getLoadDim()
        h = centerOfMass_shift * shape_width  # distance of the centroid away from the center of the long middle
        corners = np.array([[-shape_width / 2 + shape_thickness - h, -shape_thickness / 2],  # left
                            [-shape_width / 2 + shape_thickness - h, -shape_height / 2],
                            [-shape_width / 2 - h, -shape_height / 2],
                            [-shape_width / 2 - h, shape_height / 2],
                            [-shape_width / 2 + shape_thickness - h, shape_height / 2],
                            [-shape_width / 2 + shape_thickness - h, shape_thickness / 2],
                            [shape_width / 2 - shape_thickness - h, shape_thickness / 2],  # right
                            [shape_width / 2 - shape_thickness - h, shape_height / 2 * SPT_ratio],
                            [shape_width / 2 - h, shape_height / 2 * SPT_ratio],
                            [shape_width / 2 - h, -shape_height / 2 * SPT_ratio],
                            [shape_width / 2 - shape_thickness - h, -shape_height / 2 * SPT_ratio],
                            [shape_width / 2 - shape_thickness - h, -shape_thickness / 2]])

        phis = np.array([0, -np.pi / 2, np.pi, np.pi / 2, 0, np.pi / 2,
                         np.pi, np.pi / 2, 0, -np.pi / 2, np.pi, -np.pi / 2])
    #
    # if maze.shape == 'RASH':  # This is the ASymmetrical H
    #     [shape_height, shape_width, shape_thickness] = maze.getLoadDim()
    #     assymetric_h_shift = assymetric_h_shift * ResizeFactors[maze.solver][maze.size]
    #     # TODO: implement corners and phis
    #
    # if maze.shape == 'LASH':  # This is the ASymmetrical H
    #     [shape_height, shape_width, shape_thickness] = maze.getLoadDim()
    #     assymetric_h_shift = assymetric_h_shift * ResizeFactors[maze.solver][maze.size]
    #     # TODO: implement corners and phis
    return corners, phis


def init_sites(my_maze, n: int, radius=1):
    """

    :param my_maze: b2World
    :param n: number of attachment sites
    :param radius: radius of circular object
    :return: 2xn numpy matrix, with x and y coordinates of attachment sites, in the load coordinate system
    and n numpy matrix, with angles of normals of attachment sites, measured against the load coordinate system
    """
    if my_maze.shape == 'circle':
        theta = -np.linspace(0, 2 * np.pi, n)
        sites = radius * np.transpose(np.vstack([np.cos(theta), np.sin(theta)]))
        phi_default = theta
        return sites, phi_default

    else:
        corners, phis = corners_phis(my_maze)

        # the corners  in corners must be ordered like this: finding the intersection of the negative y-axis,
        # and the shape, and going clockwise find the first corner. Then go clockwise in order of the corners.
        # corners = np.vstack([[0, corners[0, 1]], corners])
        # phis = np.append(phis, phis[0])

        def linear_combination(step_size, start, end):
            return start + step_size * (end - start) / np.linalg.norm(start - end)

        # walk around the shape
        i = 1
        delta = my_maze.circumference() / n
        step_size = delta
        sites = np.array([linear_combination(np.random.uniform(0, 1), corners[0], corners[1])])
        phi_default = np.array([phis[0]])
        start = sites[-1]
        aim = corners[1]

        while sites.shape[0] < n:
            if np.linalg.norm(start - aim) > step_size:
                sites = np.vstack([sites, linear_combination(step_size, start, aim)])
                start = sites[-1]
                phi_default = np.append(phi_default, phis[(i - 1) % corners.shape[0]])
                step_size = delta

            else:
                step_size = step_size - np.linalg.norm(start - aim)
                i = i + 1
                start = corners[(i - 1) % corners.shape[0]]
                aim = corners[i % corners.shape[0]]
        return sites, phi_default
