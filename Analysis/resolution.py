from Directories import data_home, home
import pandas as pd
import numpy as np
from Setup.Maze import ResizeFactors

# resolution ...
# find out for every size of experiment, what is the noise in angle (3 exp each)
# and what is the noise in rad (* averRadius)

# What is the dependency on size?
# Give an upper boundary on the noise for XL and recalculate for size.


specific_resolution = 1


def resolution(size, solver, shape):
    if solver == 'human':
        Res = {'Large': 0.1, 'L': 0.1, 'Medium': 0.07, 'M': 0.07, 'Small Far': 0.02, 'Small Near': 0.02, 'S': 0.02}
        return Res[size] * 0.5  # had to add twice the resolution, because the fast marching method was not able to pass

    # print('My resolution is to high')
    if shape == 'SPT':
        return specific_resolution * 0.1 * ResizeFactors[solver][size]  # used to be 0.1

    return 0.1 * ResizeFactors[solver][size]


def noise(values):
    return np.abs(np.mean(values[1:] - values[:-1]))


# filenames_group = df[[d'filename', 'solver', 'maze size', 'shape']].groupby(['solver', 'maze size', 'shape'])
# columns = ['filename', 'size', 'shape', 'x noise', 'theta noise']
# df_noise = pd.DataFrame(
#     # columns=columns, index=['filename', ]
# )
#
# for (solver, size, shape), df1 in filenames_group:
#     for index in df1.index[::4]:
#         if solver != 'humanhand':
#             filename = df1['filename'].loc[index]
#             x = Get(filename, solver)
#             slice = range(int(x.position.shape[0] / 2), int(x.position.shape[0] / 2 + 20))
#
#             new = pd.DataFrame([[filename, size, shape, noise(x.position[slice, 0]), noise(x.angle[slice])]],
#                                columns=columns)
#             df_noise = df_noise.append(new, ignore_index=True)
#             print(x)
#             # _, axes = plt.subplots(num="x")
#             # axes.plot(x.position[slice, 0])
#             # _, axes = plt.subplots(num="angle")
#             # axes.plot(x.angle[slice])
#             # plt.show()
#
#             df_noise.to_json(df_dir + '.json')
#

if __name__ == '__main__':
    df = pd.read_json(data_home + 'DataFrame\\data_frame.json')
    df_dir = home + 'Analysis\\resolution_noise_exp'
# StartedScripts: resolution dependent on object size

# x = Get('M_SPT_4340004_MSpecialT_3_ants (part 1)', 'ant')
# x.play(10, 'Display')














