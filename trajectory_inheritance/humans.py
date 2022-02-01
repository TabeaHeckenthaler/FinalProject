from abc import ABC
from Box2D import b2Vec2
from Directories import MatlabFolder
import scipy.io as sio
import numpy as np
from os import path
from trajectory_inheritance.forces import Forces, sheet
from trajectory_inheritance.participants import Participants
from Setup.Maze import Maze
from PhysicsEngine.drawables import colors, Circle, Line

participant_number = {'Small Near': 1, 'Small Far': 1, 'Medium': 9, 'Large': 26}

angle_shift = {  # the number keys describe the names, zero based (Avirams numbering is 1 based)
                'Medium': {0: 0,
                           1: np.pi / 2, 2: np.pi / 2, 3: np.pi / 2,
                           4: np.pi, 5: np.pi,
                           6: -np.pi / 2, 7: -np.pi / 2, 8: -np.pi / 2},

                # the number keys describe the names, based on 0=A, ..., 25=Z.
                'Large': {0: 0, 1: 0,
                          2: np.pi / 2,
                          3: np.pi / 2, 4: np.pi / 2, 5: np.pi / 2, 6: np.pi / 2, 7: np.pi / 2, 8: np.pi / 2,
                          9: np.pi / 2,
                          10: 0,
                          11: np.pi / 2,
                          12: np.pi, 13: np.pi, 14: np.pi, 15: np.pi,
                          16: -np.pi / 2,
                          17: np.pi / 2,  # this seems to be a mistake in Avirams code
                          18: -np.pi / 2, 19: -np.pi / 2, 20: -np.pi / 2, 21: -np.pi / 2, 22: -np.pi / 2, 23: -np.pi / 2,
                          24: -np.pi / 2,
                          25: -np.pi / 2,
                          },
              }


def get_excel_worksheet_index(filename) -> int:
    """
    :param filename: filename of the tracked movie (like 'medium_20211006172352_20211006172500')
    :return: index of the excel worksheet line
    """
    number_exp = [i for i in range(1, int(sheet.dimensions.split(':')[1][1:]))
                  if sheet.cell(row=i, column=1).value is not None][-1]

    times_list = filename.split('_')[1:3]
    indices = []

    for i in range(2, number_exp + 1):
        in_filled_lines = (i <= number_exp and sheet.cell(row=i, column=1).value is not None)
        old_filename_times = sheet.cell(row=i, column=1).value.split(' ')[0].split('_')
        if len([ii for ii in range(len(times_list))
                if in_filled_lines and times_list[ii] in old_filename_times]) > 1:
            indices.append(i)

    if filename == 'medium_20201220103118_20201220110157_2':
        return 4
    if filename == 'medium_20201220103118_20201220110157':
        return 5

    if len(indices) == 1:
        return indices[0]
    elif len(times_list[-1]) > 1:  # has to be the first run
        return indices[int(np.argmin([sheet.cell(row=index, column=6).value for index in indices]))]
    elif len(times_list[-1]) == 1:
        return indices[np.argsort([sheet.cell(row=index, column=6).value
                                   for index in indices])[int(times_list[-1]) - 1]]
    elif len(indices) == 0:
        print('cant find your movie')


class Humans_Frame:
    def __init__(self, size):
        self.position = np.zeros((participant_number[size], 2))
        self.angle = np.zeros(participant_number[size])
        self.carrying = np.zeros((participant_number[size]))
        self.major_axis_length = list()
        # self.forces = list()


class Humans(Participants, ABC):
    def __init__(self, x, color=''):
        super().__init__(x, color='')

        self.excel_index = get_excel_worksheet_index(self.filename)
        self.number = len(self.gender())

        # contains list of occupied sites, where site A carries index 0 and Z carries index 25 (for size 'large').
        self.occupied = list(self.gender().keys())

        self.matlab_loading(x)
        self.angles = self.get_angles()
        self.positions = self.get_positions()
        self.gender_string = self.gender()

        if sheet.cell(row=self.excel_index, column=19).value != '/':
            self.forces = Forces(self, x)
            Participants.forcemeter = True

    def __len__(self):
        return len(self.occupied)

    def interpolate_falling_hats(self, matlab_cell):
        """
        Check if I noticed any switch:
        """
        hats_initial = set(np.array(matlab_cell[0][:, 4], dtype=int))
        if len(hats_initial) != len(self.occupied):
            raise ValueError('Your list of participants in Testable.xlxs line ' + str(self.excel_index) +
                             ' is not the same as the tracking.')

        def ranges(nums):
            nums = sorted(set(nums))
            gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
            edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
            edges = list(zip(edges, edges))
            return [(a, b+1) for (a, b) in edges]

        def switching(receiver, donor):
            frames_switch = [fr for fr in range(len(matlab_cell)) if len(np.where(matlab_cell[fr] == donor)[0]) != 0]
            for fr in frames_switch:
                matlab_cell[fr][matlab_cell[fr] == donor] = receiver

        def add_interpolation(missing_ID):
            start = matlab_cell[frame1 - 1][:, :-1][matlab_cell[frame1 - 1][:, 4] == missing_ID][0]
            if len(matlab_cell) == frame2:
                end = start  # lost frame in the last frame, so we assume the hat didn't move in the end
            else:
                end = matlab_cell[frame2][:, :-1][matlab_cell[frame2][:, 4] == missing_ID][0]

            interpolated = np.linspace(start, end, frame2 - frame1)
            for i, frame in enumerate(range(frame1, frame2)):
                matlab_cell[frame] = np.vstack([matlab_cell[frame][:missing_ID - 1],
                                                np.array([*interpolated[i], missing_ID]),
                                                matlab_cell[frame][missing_ID - 1:]])

        def delete(hat):
            matlab_cell[frame] = np.delete(matlab_cell[frame],
                                           np.where(matlab_cell[frame][:, 4] == hat)[0],
                                           0)
        for hat in hats_initial:

            switch_frames = [i for i, frame in enumerate(matlab_cell) if hat not in frame[:, 4]]
            for frame1, frame2 in ranges(switch_frames):
                flying_hats = sheet.cell(row=self.excel_index, column=20).value
                if flying_hats is not None and int(flying_hats.split(':')[0]) == hat:
                    for new_hat in flying_hats.split(':')[1].split(', '):
                        switching(hat, int(new_hat))

            interpolate_frames = [i for i, frame in enumerate(matlab_cell) if hat not in frame[:, 4]]
            for frame1, frame2 in ranges(interpolate_frames):
                add_interpolation(hat)

        new_hats_frames = [i for i, frame in enumerate(matlab_cell) if frame.shape != (len(self.occupied), 5)]
        for frame in new_hats_frames:
            additional_hats = set(np.array(matlab_cell[frame][:, 4], dtype=int)) - hats_initial
            for hat in additional_hats:
                delete(hat)
        return matlab_cell

    def matlab_loading(self, x) -> None:
        file = sio.loadmat(MatlabFolder(x.solver, x.size, x.shape) + path.sep + x.filename)
        matlab_cell = np.squeeze(file['hats'])

        self.interpolate_falling_hats(matlab_cell)
        my_maze = Maze(x)

        Medium_id_correction_dict = {1: 1, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2}

        for i, Frame in enumerate(matlab_cell):
            data = Frame

            # to sort the data
            humans_frame = Humans_Frame(self.size)

            # if identities are given
            if data.shape[1] > 4:
                data = data[data[:, 4].argsort()]

            if x.size in ['Medium', 'Large']:
                if x.filename == 'large_20210419100024_20210419100547':
                    for false_reckog in [8., 9.]:
                        index = np.where(data[:, 4] == false_reckog)
                        data = np.delete(data, index, 0)

                # correct the wrong hat identities
                if x.size == 'Medium':
                    data = data[np.vectorize(Medium_id_correction_dict.get)(data[:, 4]).argsort()]
                    data[:, 4] = np.vectorize(Medium_id_correction_dict.get)(data[:, 4])

                if humans_frame.position[self.occupied].shape != data[:, 2:4].shape:
                    k = 1
                humans_frame.position[self.occupied] = data[:, 2:4] + np.array([x.x_error, x.y_error])

                # if force meters were installed, then only carrying boolean and angle were included in .mat file
                if data.shape[1] > 5:
                    humans_frame.carrying[self.occupied] = data[:, 5]

                    # angle to force meter
                    if x.filename not in ['medium_20201223130749_20201223131147',
                                          'medium_20201223125622_20201223130532']:
                        # here, the identities were given wrong in the tracking
                        humans_frame.angle[self.occupied] = data[:, 6] * np.pi / 180 + x.angle_error
                    else:
                        humans_frame.angle[self.occupied] = data[:, 6] * np.pi / 180 + x.angle_error
                        my_maze.set_configuration(x.position[i], x.angle[i])
                        humans_frame.angle[self.occupied] = self.angle_to_forcemeter(humans_frame.position, my_maze,
                                                                                     x.angle[i], x.size)

            self.frames.append(humans_frame)
        return

    def angle_to_forcemeter(self, positions, my_maze, angle, size) -> np.ndarray:
        r = positions[self.occupied] - my_maze.force_attachment_positions()[self.occupied]
        angles_to_normal = np.arctan2(r[:, -1], r[:, 0]) - \
                           np.array([angle_shift[size][occ] for occ in self.occupied]) - angle
        return angles_to_normal

    def get_angles(self) -> np.ndarray:
        return np.array([fr.angle for fr in self.frames])

    def get_positions(self) -> np.ndarray:
        return np.array([fr.position for fr in self.frames])

    def averageCarrierNumber(self) -> int:
        return self.number

    def gender(self) -> dict:
        """
        return dict which gives the gender of every participant. The keys of the dictionary are indices_to_coords of participants,
        where participant A has index 0, B has index 1, ... and Z has index 25. This is different from the counting in
        Aviram's movies!!
        """
        gender_string = list(sheet.cell(row=self.excel_index, column=17).value)

        if len(gender_string) != participant_number[self.size] \
                and not (len(gender_string) in [1, 2] and self.size == 'Medium'):
            print('you have an incorrect gender string in ' + str(self.excel_index))
        return {i: letter for i, letter in enumerate(gender_string) if letter != '0'}

    def draw(self, display) -> None:
        for part in self.occupied:
            Circle(self.positions[display.i, part], 0.1, colors['hats'], hollow=False).draw(display)
            text = display.font.render(str(part), True, colors['text']) # TODO: Tabea, do this!
            display.screen.blit(text, display.m_to_pixel(self.positions[display.i, part]))
            if hasattr(self, 'forces'):
                force_attachment = display.my_maze.force_attachment_positions()
                Circle(force_attachment[part], 0.05, (0, 0, 0), hollow=False).draw(display)
                # Line(self.positions[display.i, force_vector], force_attachment[force_vector], (0, 0, 0)).draw(display)
