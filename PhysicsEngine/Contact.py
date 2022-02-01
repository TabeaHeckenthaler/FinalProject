from scipy.spatial import cKDTree
from Setup.MazeFunctions import BoxIt
import numpy as np
from Setup.Load import loops
from Analysis.GeneralFunctions import flatten
import itertools

# maximum distance between fixtures to have a contact (in cm)
distance_upper_bound = 0.04


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def possible_configuration(load, maze_corners, former_found) -> tuple:
    """
    this function takes a list of corners (lists have to list rectangles in sets of 4 corners)
    and checks, whether a rectangle from the load overlaps with a rectangle from the maze boundary
    :return: bool, whether the shape intersects the maze
    """
    i, ii = former_found
    load_corners = np.array(flatten(loops(load)))
    loop_indices = [0, 1, 2, 3, 0]
    for load_vertices_list in np.array_split(load_corners, int(load_corners.shape[0]/4)):
        for maze_vertices_list in maze_corners:

            if intersect(load_vertices_list[i], load_vertices_list[loop_indices[i + 1]],
                         maze_vertices_list[ii], maze_vertices_list[loop_indices[ii + 1]]):
                return False, (i, ii)

            for i, ii in itertools.product(range(len(loop_indices)-1), range(len(loop_indices)-1)):
                if intersect(load_vertices_list[loop_indices[i]], load_vertices_list[loop_indices[i + 1]],
                             maze_vertices_list[loop_indices[ii]], maze_vertices_list[loop_indices[ii + 1]]):
                    return False, (i, ii)
    return True, (0, 0)
    # return np.any([f.TestPoint(load.position) for f in maze.body.fixtures])


def contact_loop_experiment(load, maze) -> list:
    """
    :return: list of all the points in world coordinates where the load is closer to the maze than distance_upper_bound.
    """
    edge_points = contact = []
    load_vertices = loops(load)

    for load_vertice in load_vertices:
        edge_points = edge_points + BoxIt(load_vertice, distance_upper_bound).tolist()

    load_tree = cKDTree(edge_points)
    in_contact = load_tree.query(maze.slitTree.data, distance_upper_bound=distance_upper_bound)[1] < \
                 load_tree.data.shape[0]

    if np.any(in_contact):
        contact = contact + maze.slitTree.data[np.where(in_contact)].tolist()
    return contact
