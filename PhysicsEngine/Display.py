import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_SPACE)
from PhysicsEngine.drawables import colors
import os
import numpy as np
import cv2
import sys
from Directories import video_directory
# from Video_Editing.merge_videos import merge_frames
from os import path

try:
    from mayavi import mlab
except:
    pass


class Display:
    def __init__(self, name: str, my_maze, wait=0, ps=None, i=0, videowriter=False, config=None):
        self.my_maze = my_maze
        self.filename = name
        self.ppm = int(1100 / self.my_maze.arena_length)  # pixels per meter
        self.height = int(self.my_maze.arena_height * self.ppm)
        self.width = 1100

        pygame.font.init()  # display and fonts
        self.font = pygame.font.Font('freesansbold.ttf', 25)
        # self.monitor = {'left': 0, 'top': 0,
        #                 'width': int(Tk().winfo_screenwidth() * 0.9), 'height': int(Tk().winfo_screenheight() * 0.8)}
        self.monitor = {'left': 0, 'top': 0,
                        'width': self.width, 'height': self.height}
        self.screen = self.create_screen()
        self.arrows = []
        self.circles = []
        self.polygons = []
        self.points = []
        self.wait = wait
        self.i = i

        if config is not None:
            my_maze.set_configuration(config[0], config[1])

        self.renew_screen()
        self.ps = ps
        if videowriter:
            if self.ps is not None:
                self.VideoShape = (max(self.height, mlab.screenshot(self.ps.fig, mode='rgb').shape[0]),
                                   self.width + mlab.screenshot(self.ps.fig, mode='rgb').shape[1])

            else:
                self.VideoShape = (self.monitor['height'], self.monitor['width'])
            self.VideoWriter = cv2.VideoWriter(path.join(video_directory, sys.argv[0].split('/')[-1].split('.')[0] + '.mp4v'),
                                               cv2.VideoWriter_fourcc(*'DIVX'), 20,
                                               (self.VideoShape[1], self.VideoShape[0]))

    def create_screen(self, caption=str(), free=False) -> pygame.surface:
        pygame.font.init()  # display and fonts
        pygame.font.Font('freesansbold.ttf', 25)

        if free:  # screen size dependent on trajectory_inheritance
            position = None  # TODO
            self.ppm = int(1000 / (np.max(position[:, 0]) - np.min(position[:, 0]) + 10))  # pixels per meter
            self.width = int((np.max(position[:, 0]) - np.min(position[:, 0]) + 10) * self.ppm)
            self.height = int((np.max(position[:, 1]) - np.min(position[:, 1]) + 10) * self.ppm)

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.monitor['left'], self.monitor['top'])
        screen = pygame.display.set_mode((self.width, self.height), 0, 32)
        if self.my_maze is not None:
            pygame.display.set_caption(self.my_maze.shape + ' ' + self.my_maze.size + ' ' + self.my_maze.solver + ': ' + caption)
        return screen

    def m_to_pixel(self, r):
        return [int(r[0] * self.ppm), self.height - int(r[1] * self.ppm)]

    def update_screen(self, x, i):
        self.i = i
        if self.wait > 0:
            pygame.time.wait(int(self.wait))
        self.draw(x)
        end = self.keyboard_events()
        return end

    def renew_screen(self, frame=0, movie_name=None):
        self.screen.fill(colors['background'])

        self.drawGrid()
        self.polygons = self.circles = self.points = self.arrows = []

        if frame is not None:
            text = self.font.render(movie_name, True, colors['text'])
            text_rect = text.get_rect()
            text2 = self.font.render('Frame: ' + str(self.i), True, colors['text'])
            self.screen.blit(text2, [0, 25])
            self.screen.blit(text, text_rect)

    def end_screen(self):
        if hasattr(self, 'VideoWriter'):
            self.VideoWriter.release()
            print('Saved Movie in ', path.join(video_directory, sys.argv[0].split('/')[-1].split('.')[0] + '.mp4v'))
        # if self.ps is not None:
        #     self.ps.VideoWriter.release()
        pygame.display.quit()

    def pause_me(self):
        pygame.time.wait(int(100))
        events = pygame.event.get()
        for event in events:
            if event.type == KEYDOWN and event.key == K_SPACE:
                return
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.end_screen()
                return
        self.pause_me()

    def draw(self, x):
        self.my_maze.draw(self)
        if self.ps is not None:
            if self.i <= 1 or self.i >= len(x.angle)-1:
                kwargs = {'color': (0, 0, 0), 'scale_factor': 1.}
            else:
                kwargs = {}
            self.ps.draw(x.position[self.i:self.i + 1], x.angle[self.i:self.i + 1], **kwargs)
        if hasattr(x, 'participants'):
            if hasattr(x.participants, 'forces'):
                x.participants.forces.draw(self, x)
            if hasattr(x.participants, 'positions'):
                x.participants.draw(self)
        self.display()
        # self.write_to_Video()
        return

    def display(self):
        pygame.display.flip()

    def write_to_Video(self):
        if hasattr(self, 'VideoWriter'):
            pass
            img = np.swapaxes(pygame.surfarray.array3d(self.screen), 0, 1)
            if hasattr(self, 'ps'):
                img = merge_frames([img, mlab.screenshot(self.ps.fig, mode='rgb')],
                                   (self.VideoShape[0], self.VideoShape[1], 3),
                                   [[0, 0], [0, self.width]])

            self.VideoWriter.write(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))

        # if self.ps is not None:
        #     self.ps.write_to_Video()

    def draw_contacts(self, contact):
        for contacts in contact:
            pygame.draw.circle(self.screen, colors['contact'],  # On the corner
                               [int(contacts[0] * self.ppm),
                                int(self.height - contacts[1] * self.ppm)],
                               10,
                               )

    def drawGrid(self):
        block = 2
        block_size = 2 * self.ppm
        for y in range(np.int(np.ceil(self.height / self.ppm / block) + 1)):
            for x in range(np.int(np.ceil(self.width / self.ppm / block))):
                rect = pygame.Rect(x * block_size, self.height -
                                   y * block_size, block_size, block_size)
                pygame.draw.rect(self.screen, colors['grid'], rect, 1)

    def keyboard_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == K_ESCAPE):  # you can also add 'or Finished'
                # The user closed the window or pressed escape
                self.end_screen()
                return True

            if event.type == KEYDOWN and event.key == K_SPACE:
                self.pause_me()
            # """
            # To control the frames:
            # 'D' = one frame forward
            # 'A' = one frame backward
            # '4' (one the keypad) = one second forward
            # '6' (one the keypad) = one second backward
            # """
            # if event.key == K_a:
            #     i -= 1
            # elif event.key == K_d:
            #     i += 1
            # elif event.key == K_KP4:
            #     i -= 30
            # elif event.key == K_KP6:
            #     i += 30

    def snapshot(self, filename, *args):
        pygame.image.save(self.screen, filename)
        if 'inlinePlotting' in args:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.set_size_inches(30, 15)
            img = mpimg.imread(filename)
            plt.imshow(img)
        return



