
import numpy as np
from Box2D import b2Vec2

from Setup.Load import loops
from Setup.MazeFunctions import ClosestCorner


def Forces_old(my_load, my_maze, **kwargs):
    grC = 1
    gravCenter = np.array([my_maze.arena_length * grC, my_maze.arena_height / 2])  # this is the 'far away point', to which the load gravitates

    load_vertices = loops(my_load)

    """ Where Force attaches """
    ForceAttachments = [ClosestCorner(load_vertices, gravCenter)]

    """ Magnitude of forces """
    arrows = []
    for ForceAttachment in ForceAttachments:
        # f_x = -rd.gauss(x.xForce * (ForceAttachment[0] - maze.arena_length * grC) / maze.arena_length * grC,
        #                 x.xDev)
        # f_y = -rd.gauss(x.yForce * (load.position.y - maze.arena_height / 2) / maze.arena_height / 2, x.yDev)

        f_x = 1
        f_y = 0

        my_load.ApplyForce(b2Vec2([f_x, f_y]),
                           # b2Vec2(ForceAttachment),
                           my_load.position,
                           True)

        start = ForceAttachment
        end = ForceAttachment + [f_x, f_y]
        arrows.append((start, end, ''))

    return ForceAttachments, arrows