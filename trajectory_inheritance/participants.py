from abc import abstractmethod
from PhysicsEngine.drawables import Drawables, colors
import pygame
import numpy as np


class Participants(Drawables):
    def __init__(self, x, color=colors['puller']):
        super().__init__(color)
        self.filename = x.filename
        self.frames = list()
        self.size = x.size
        self.VideoChain = x.VideoChain
        self.positions = np.array([])

    @abstractmethod
    def matlab_loading(self, x) -> None:
        pass

    @abstractmethod
    def averageCarrierNumber(self) -> float:
        pass

    def draw(self, display) -> None:
        for part in range(self.positions.shape[1]):
            pygame.draw.circle(display.screen, self.color, display.m_to_pixel(self.positions[display.i, part, 0]), 7.)
