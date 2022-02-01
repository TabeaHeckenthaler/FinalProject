from matplotlib import pyplot as plt
import numpy as np

# def detect_loops(self):
#     path = self.generate_path(length=40, ind=True)[0]
#     for ind in path:
#         if np.sum(path == ind) > 4:
#             return True


def plot_distances(self, index=0, plane='theta'):
    ex = self.conf_space.extent
    X, Y, Theta = np.meshgrid(np.linspace(ex['x'][0], ex['x'][1], self.conf_space.space.shape[0]),
                              np.linspace(ex['y'][0], ex['y'][1], self.conf_space.space.shape[1]),
                              np.linspace(ex['theta'][0], ex['theta'][1], self.conf_space.space.shape[2]),
                              indexing='ij')
    if plane == 'x':
        plt.contour(Y[index, :, :], Theta[index, :, :], self.distance[index, :, :],
                    [0], colors='black', linewidths=3)
        plt.contour(Y[index, :, :], Theta[index, :, :], self.distance[index, :, :])
        plt.xlabel('y')
        plt.ylabel('theta')
    elif plane == 'y':
        plt.contour(X[:, index, :], Theta[:, index, :], self.distance[:, index, :],
                    [0], colors='black', linewidths=3)
        plt.contour(X[:, index, :], Theta[:, index, :], self.distance[:, index, :])
        plt.xlabel('x')
        plt.ylabel('theta')
    elif plane == 'theta':
        plt.contour(X[:, :, index], Y[:, :, index], self.distance[:, :, index],
                    [0], colors='black', linewidths=3)
        plt.contour(X[:, :, index], Y[:, :, index], self.distance[:, :, index])
        plt.xlabel('x')
        plt.ylabel('y')
    plt.colorbar()
    plt.show(block=False)

    # plt.imshow(self.distance[:, index, :]) (if you have the mask in self.distance it looks even better)
    return


voxel = np.array([[[False, False, False],
                   [False, True, False],
                   [False, False, False]],
                  [[False, True, False],
                   [True, False, True],
                   [False, True, False]],
                  [[False, False, False],
                   [False, True, False],
                   [False, False, False]]])
