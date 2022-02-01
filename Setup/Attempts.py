import pandas as pd
from scipy.ndimage import median_filter
import numpy as np
from Setup.MazeFunctions import extend
from Setup.Maze import Maze
from Setup.Load import loops
from trajectory_inheritance.trajectory import get

smoothing_window = 6


def Attempt_setup(x, my_load, my_attempt_zone, starting_line, *args):
    load_vertices = loops(my_load)  # [0]

    inside = False
    for zone_fixture in my_attempt_zone.fixtures:
        for vertices in load_vertices:
            inside = inside or zone_fixture.TestPoint(vertices)
    if inside:
        x_distance = min([starting_line - x_coor for x_coor in [p[0] for p in load_vertices]])
        x = extend(x, 'start', x_distance, *args)
    return x


def Attempt_loop(finish_line, my_attempt_zone, my_load, interval=1, **kwargs):
    load_vertices = loops(my_load)
    inside = False
    for zone_fixture in my_attempt_zone.fixtures:
        for vertices in load_vertices:
            inside = inside or zone_fixture.TestPoint(vertices)

        if 'Caption' in kwargs:
            addition = 'I cut off experiments once all the points crossed through the slit'
            kwargs['Caption'] + addition

    if all([x_coor > finish_line for x_coor in [p[0] for p in load_vertices]]):  # this means: we wone!
        inside = False

    if 'Caption' in kwargs:
        addition = 'Counted as attempt if any corner of the shape is found in the AttemptZone in given frame'
        kwargs['Caption'] + addition
    return [inside for i in range(interval)]


def AttemptZoneDim(solver, shape, size, my_maze):
    shape_thickness = getLoadDim(solver, shape, size)[2]
    return shape_thickness + my_maze.wallthick, my_maze.exit_size


def AddAttemptZone(my_maze, x, **kwargs):
    from Box2D import b2BodyDef, b2_staticBody, b2Vec2
    my_attempts_zone = my_maze.CreateBody(b2BodyDef(position=(my_maze.slits[-1] + my_maze.wallthick,
                                                              my_maze.arena_height / 2),
                                                    type=b2_staticBody,
                                                    fixedRotation=True))

    my_attempts_zone.userData = 'my_attempt_zone'

    x_size, y_size = AttemptZoneDim(x.solver, x.shape, x.size, my_maze)

    my_attempts_zone.CreatePolygonFixture(vertices=[
        (0, y_size / 2),
        (0, -y_size / 2),
        (-x_size, -y_size / 2),
        (-x_size, y_size / 2)],
        density=1, friction=0, restitution=0,
    )

    my_attempts_zone.CreatePolygonFixture(vertices=[
        (5, my_maze.arena_height / 2),
        (5, -my_maze.arena_height / 2),
        (my_maze.wallthick / 2, -my_maze.arena_height / 2),
        (my_maze.wallthick / 2, my_maze.arena_height / 2)],
        density=1, friction=0, restitution=0,
    )

    # this is necessary because I don't want the left part of the shape to be
    # before the zone, and the last part of the shape to be after the zone. So, I have to extend the zone.

    my_attempts_zone.CreateCircleFixture(radius=x_size - my_maze.wallthick / 2,
                                         pos=b2Vec2(-my_maze.wallthick / 2, -y_size / 2))
    my_attempts_zone.CreateCircleFixture(radius=x_size - my_maze.wallthick / 2,
                                         pos=b2Vec2(-my_maze.wallthick / 2, y_size / 2))

    if 'Caption' in kwargs:
        addition = 'Attempt zone is a capsule.'
        kwargs['Caption'] + addition

    for fix in my_attempts_zone.fixtures:
        fix.sensor = True

    return my_attempts_zone


def Attempts(x, *args, **kwargs):
    return None
    # TODO
    window = x.fps * smoothing_window
    speed = 1
    # interval = 10

    if x.shape == 'SPT':
        x, attempts = x, [True for i in range(len(x.frames))]
    else:
        x, attempts = x.play('attempt', *args, **kwargs)[:len(x.frames)]

    attempts_smoothed = median_filter(attempts, size=window)
    if 'Caption' in kwargs:
        addition = 'Median filter with window ' + str(smoothing_window) + ' s, when separating attempts '
        kwargs['Caption'] + addition

    at = pd.DataFrame({"a": attempts_smoothed})

    no_attempts = at.a.cumsum()[~at.a].reset_index().groupby('a')['index'].agg(['first', 'last']).values.tolist()

    if len(no_attempts) == 0:  # meaning, the entire run is counted as attempt
        no_attempts = [[-1, -1]]

    attempts_smoothed = [[no_attempts[0][-1] + 1, len(x.frames)]]  # first attempt starts, when last Non Attempt ends
    for NoAttempt in no_attempts[1:]:  # if we have multiple attempts
        attempts_smoothed[-1][-1] = NoAttempt[0]
        attempts_smoothed.append([NoAttempt[1] + 1, len(x.frames)])
    # attempts = np.array([attempt for attempt in attempts if attempt[-1] - attempt[0] > window])
    return attempts_smoothed


def PlayAttempts(x, speed, *args, **kwargs):
    at = Attempts(x)
    print(x.filename)
    for i in range(len(at)):
        print('frames: ' + str(at[i]) + '  in  ' + str((int(at[i][1]) - int(at[i][0])) / x.fps) + 's')
        x.play(speed, 'attempt', indices=at[i], *args, **kwargs)
        if i < len(at) - 1:
            breakpoint()


def AttemptNumber(x, *args, **kwargs):
    a = Attempts(x, 'extend', *args, **kwargs)
    return len(a), [['Number of Attempts per trajectory_inheritance', 'NumAttempts []']]


def AttemptDuration(x, *args, **kwargs):
    durations = list()
    a = Attempts(x, 'extend', *args, **kwargs)
    for at in a:
        durations.append((at[-1] - at[0]) / x.fps)
    return np.sum(durations), [['Duration per attempt', 'Duration/exit_size [s/cm]']]


if __name__ == '__main__':
    solver = 'human'
    x = get('medium_20201221135753_20201221140218')
    attempts = [False]
    my_maze = Maze(size=x.size, shape=x.shape, solver=x.solver, position=x.position[0], angle=x.angle[0])
    my_attempt_zone = AddAttemptZone(my_maze, x)

    starting_line = -AttemptZoneDim(x.solver, x.shape, x.size, my_maze)[0] + my_maze.slits[0]
    finish_line = my_maze.slits[-1] + my_maze.wallthick

    # for i in range(10):
    #     attempts = attempts + Attempt_loop(finish_line, my_attempt_zone, load, **kwargs)
    #     kwargs['attempt'] = attempts[-1]

    x = Attempt_setup(x, my_maze.bodies[-1], my_attempt_zone, starting_line)