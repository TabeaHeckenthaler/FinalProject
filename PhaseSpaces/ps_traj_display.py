from trajectory_inheritance.trajectory import get
from PhaseSpaces.PhaseSpace import PhaseSpace
from DataFrame.dataFrame import myDataFrame
import seaborn as sns
import numpy as np

size, shape, solver = 'M', 'H', 'ant'

ps = PhaseSpace(solver=solver, size=size, shape=shape, name=size + '_' + shape)
ps.load_space()
ps.visualize_space()

exps = myDataFrame[(myDataFrame['size'] == size) & (myDataFrame['shape'] == shape) & (myDataFrame['solver'] == solver)]
filenames = exps['filename'][:10]
trajs = [get(filename) for filename in filenames]
trajs[1].play(ps=ps, step=20, videowriter=True)
cmap = sns.color_palette("rocket_r", as_cmap=True)
colors = cmap(np.linspace(0.2, 1, len(trajs)))[:, :3]

for traj, color in zip(trajs[3:5], [colors[3], colors[9]], ):
    ps.draw(traj.position, traj.angle, scale_factor=0.2, color=tuple(color))

k = 1

# trajs[1].play(ps=ps, step=20, videowriter=True)
#
# # for M
# i = 6
# ps.draw(trajs[i].position, trajs[i].angle, scale_factor=0.3, color=tuple(colors[i]))
#
# # for XL
# i = 1
# ps.draw(trajs[i].position, trajs[i].angle, scale_factor=0.3, color=tuple(colors[i]))