from os import path, mkdir
from datetime import datetime

# home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
# data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)
home = path.join(path.abspath(__file__).split('\\')[0]+path.sep, *path.abspath(__file__).split(path.sep)[1:-1])
data_home = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'PyCharm_Data', 'AntsShapes')

work_dir = path.join(data_home, 'Pickled_Trajectories')
SaverDirectories = {'ant': path.join(work_dir, 'Ant_Trajectories'),
                    'human': path.join(work_dir, 'Human_Trajectories'),
                    'humanhand': path.join(work_dir, 'HumanHand_Trajectories'),
                    'gillespie': path.join(work_dir, 'Gillespie_Trajectories'),
                    'ps_simulation': path.join(work_dir, 'PS_simulation_Trajectories')}

mini_work_dir = path.join(data_home, 'mini_Pickled_Trajectories')
mini_SaverDirectories = {'ant': path.join(mini_work_dir, 'Ant_Trajectories'),
                         'human': path.join(mini_work_dir, 'Human_Trajectories'),
                         'humanhand': path.join(mini_work_dir, 'HumanHand_Trajectories'),
                         'gillespie': path.join(mini_work_dir, 'Gillespie_Trajectories'),
                         'ps_simulation': path.join(mini_work_dir, 'PS_simulation_Trajectories')}

PhaseSpaceDirectory = path.join(data_home, 'PhaseSpaces')

excel_sheet_directory = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'Human Experiments')
contacts_dir = path.join(data_home, 'Contacts', 'ant')
df_dir = path.join(data_home, 'DataFrame', 'data_frame.json')
network_dir = path.join(home, 'Analysis', 'PathPy', 'Network_Images')
maze_dimension_directory = path.join(home, 'Setup')


def SetupDirectories():
    if not (path.isdir(SaverDirectories['ant'])):
        if not path.isdir('\\\\' + SaverDirectories['ant'].split('\\')[2]):
            return
        mkdir(SaverDirectories['ant'])
    if not (path.isdir(SaverDirectories['human'])):
        mkdir(SaverDirectories['human'])
    if not (path.isdir(SaverDirectories['humanhand'])):
        mkdir(SaverDirectories['humanhand'])
    if not (path.isdir(SaverDirectories['ps_simulation'])):
        mkdir(SaverDirectories['ps_simulation'])

    if not (path.isdir(mini_SaverDirectories['ant'])):
        if not path.isdir('\\\\' + mini_SaverDirectories['ant'].split('\\')[2]):
            return
        mkdir(mini_SaverDirectories['ant'])
    if not (path.isdir(mini_SaverDirectories['human'])):
        mkdir(mini_SaverDirectories['human'])
    if not (path.isdir(mini_SaverDirectories['humanhand'])):
        mkdir(mini_SaverDirectories['humanhand'])
    if not (path.isdir(mini_SaverDirectories['ps_simulation'])):
        mkdir(mini_SaverDirectories['ps_simulation'])
    return

video_directory = path.join(home, 'Videos')
if not path.exists(video_directory):
    mkdir(video_directory)

trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedHumanMovieDirectory = path.join(excel_sheet_directory, 'Output')
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'  # TODO


def MatlabFolder(solver, size, shape):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        return path.join(trackedAntMovieDirectory, 'Slitted', shape_folder_naming[shape], size, 'Output Data')

    if solver == 'human':
        return path.join(trackedHumanMovieDirectory, size, 'Data')

    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


def NewFileName(old_filename: str, solver: str, size: str, shape: str, expORsim: str) -> str:
    import glob
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        return size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            return filename.replace(old_filename.split('_')[0], size + '_' + shape)

        else:
            if solver == 'ant':
                if size + shape in filename or size + '_' + shape in filename:
                    return filename.replace(size + shape, size + '_' + shape)
                else:
                    raise ValueError('Your filename does not seem to be right.')
            elif solver == 'human':
                return filename


SetupDirectories()
