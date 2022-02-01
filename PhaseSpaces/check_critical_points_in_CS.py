from PhaseSpaces import PhaseSpace
from copy import copy


def tunnel_center(size: str):
    if size == 'S':
        return 232, 70, 150
    else:
        return 147, 63, 150


def disconnect_center(size: str):
    if size == 'S':
        return 270, 118, 168
    else:
        print('mask around disconnect not ready')
        return 147, 63, 150


def mask_around_center(conf_space: PhaseSpace, center_function=tunnel_center):
    radiusx, radiusy, radiusz = 10, 13, 13
    mask = conf_space.empty_space()
    center = center_function(conf_space.size)
    mask[center[0] - radiusx:center[0] + radiusx,
         center[1] - radiusy:center[1] + radiusy,
         center[2] - radiusz:center[2] + radiusz] = True
    return mask


if __name__ == '__main__':
    solver, shape = 'ant', 'SPT'
    for size in ['L', 'S']:
        conf_space = PhaseSpace.PhaseSpace(solver, size, shape, name='')

        conf_space.load_space()
        conf_space.visualize_space()
        #
        # for center in [tunnel_center, disconnect_center]:
        #     mask = mask_around_center(conf_space, center)
        #     conf_space.calculate_space(mask=mask)
        #     new_space = copy(conf_space.space)
        #     conf_space.visualize_space(space=new_space)
        #     conf_space.visualize_space(space=mask, colormap='Oranges')
        #
        DEBUG = 1

        # conf_space.calculate_space()
        # conf_space.calculate_boundary()
        # conf_space.save_space()

# conf_space_labeled = PhaseSpace.PhaseSpace_Labeled(conf_space)
# conf_space_labeled.load_space()
# # conf_space_labeled.save_labeled()
#
# x = get('XL_SPT_dil9_sensing4')
# labels = States(conf_space_labeled, x, step=x.fps)
# k = 1




