from itertools import groupby
import numpy as np

states = ['0', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'j']

forbidden_transition_attempts = ['be', 'bf', 'bg',
                                 'di',
                                 'eb', 'ei',
                                 'fb', 'fi',
                                 'gf', 'ge', 'gj', 'gb',
                                 'id', 'ie', 'if']

allowed_transition_attempts = ['ab', 'ad',
                               'ba',
                               'de', 'df', 'da',
                               'ed', 'eg',
                               'fd', 'fg',
                               'gf', 'ge', 'gj',
                               'ij',
                               'jg', 'ji']


class States:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """
    def __init__(self, conf_space_labeled, x, step: int = 1):
        """
        :param step: after how many frames I add a label to my label list
        :param x: trajectory
        :return: list of strings with labels
        """
        self.time_step = step/x.fps
        indices = [conf_space_labeled.coords_to_indices(*coords) for coords in x.iterate_coords(step=step)]
        self.time_series = [conf_space_labeled.space_labeled[index] for index in indices]
        self.interpolate_zeros()
        self.state_series = self.calculate_state_series()

        if len(self.forbidden_attempts()) > 0:
            print('forbidden_attempts:', self.forbidden_attempts(), 'in', x.filename)

            # print('You might want to decrease your step size, because you might be skipping state transitions.')

    @staticmethod
    def combine_transitions(state_series) -> list:
        """
        I want to combine states, that are [.... 'gb' 'bg'...] to [... 'gb'...]
        :param state_series: series to be mashed
        :return: state_series with combined transitions
        """
        state_series = [''.join(sorted(state)) for state in state_series]
        mask = [True] + [sorted(state1) != sorted(state2) for state1, state2 in zip(state_series, state_series[1:])]
        return np.array(state_series)[mask].tolist()

    @staticmethod
    def cut_at_end(time_series) -> list:
        """
        After state 'j' appears, cut off series
        :param state_series: series to be mashed
        :return: state_series with combined transitions
        """
        if 'j' not in time_series:
            return time_series
        first_appearance = np.where(np.array(time_series) == 'j')[0][0]
        return time_series[:first_appearance+1]

    def interpolate_zeros(self) -> None:
        """
        Interpolate over all the states, that are not inside Configuration space (due to the computer representation of
        the maze not being exactly the same as the real maze)
        :return:
        """
        if self.time_series[0] == '0':
            self.time_series[0] = [l for l in self.time_series if l != '0'][:1000][0]
        for i, l in enumerate(self.time_series):
            if l == '0':
                self.time_series[i] = self.time_series[i - 1]

    def forbidden_attempts(self) -> list:
        """
        Check whether the permitted transitions are all allowed
        :return: boolean, whether all transitions are allowed
        """
        allowed = {el[0]: [] for el in allowed_transition_attempts}
        [allowed[origin].append(goal) for [origin, goal] in allowed_transition_attempts]
        # TODO
        # return [str(l0) + ' to ' + str(l1) for l0, l1 in zip(self.time_series, self.time_series[1:])
        #         if l1 not in allowed[l0]]
        return []

    def calculate_state_series(self):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in self.time_series])]
        return labels
