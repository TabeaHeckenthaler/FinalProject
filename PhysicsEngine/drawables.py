import math
import pygame

colors = {'maze': (0, 0, 0),
          'load': (250, 0, 0),
          'my_attempt_zone': (0, 128, 255),
          'text': (0, 0, 0),
          'background': (250, 250, 250),
          'hats': (51, 255, 51),
          'grid': (220, 220, 220),
          'arrow': (135, 206, 250),
          'participants': (0, 0, 0),
          'puller': (0, 250, 0),
          'lifter': (0, 250, 0),
          'empty': (0, 0, 0),
          }


class Drawables:
    def __init__(self, color):
        self.color = color

    def draw(self, display) -> None:
        pass


class Polygon(Drawables):
    def __init__(self, vertices, color='arrow'):
        super().__init__(color)
        self.vertices = vertices

    def draw(self, display):
        lines = [(vertice[0] * display.ppm, display.height - vertice[1] * display.ppm) for vertice in self.vertices]
        pygame.draw.lines(display.screen, self.color, True, lines, 3)


class Arrow(Drawables):
    def __init__(self, start, end, name=str(), color='arrow'):
        super().__init__(colors[color])
        self.start = start
        self.end = end
        self.name = name

    def draw(self, display) -> None:
        start = display.m_to_pixel(self.start)
        end = display.m_to_pixel(self.end)
        rad = math.pi / 180
        thickness, trirad = int(0.05 * display.ppm), int(0.2 * display.ppm)
        arrow_width = 150

        pygame.draw.line(display.screen, self.color, start, end, thickness)
        pygame.draw.circle(display.screen, self.color, start, 5)

        rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
        pygame.draw.polygon(display.screen, self.color, ((end[0] + trirad * math.sin(rotation),
                                                          end[1] + trirad * math.cos(rotation)),
                                                         (end[0] + trirad * math.sin(rotation - arrow_width * rad),
                                                          end[1] + trirad * math.cos(rotation - arrow_width * rad)),
                                                         (end[0] + trirad * math.sin(rotation + arrow_width * rad),
                                                          end[1] + trirad * math.cos(rotation + arrow_width * rad))))

        if self.name not in ['puller', 'lifter', 'empty']:
            text = display.font.render(str(self.name), True, colors['text'])
            display.screen.blit(text, end)


class Point(Drawables):
    def __init__(self, center, color=colors['text']):
        super().__init__(color)
        self.center = center

    def draw(self, display):
        pygame.draw.circle(display.screen, colors['text'],
                           [int(self.center[0] * display.ppm), display.height - int(self.center[1] * display.ppm)], 5)


class Circle(Drawables):
    def __init__(self, center, radius, color='arrow', hollow=True):
        super().__init__(color)
        self.center = center
        self.radius = radius
        self.hollow = hollow

    def draw(self, display) -> None:
        pygame.draw.circle(display.screen, self.color, display.m_to_pixel(self.center), int(self.radius * display.ppm))
        if self.hollow:
            pygame.draw.circle(display.screen, colors['background'],
                               display.m_to_pixel(self.center), int(self.radius * display.ppm) - 3)


class Line(Drawables):
    def __init__(self, start, end, color=colors['text']):
        super().__init__(color)
        self.start = start
        self.end = end

    def draw(self, display) -> None:
        pygame.draw.line(display.screen, self.color, display.m_to_pixel(self.start),
                         display.m_to_pixel(self.end), width=4)
