import numpy as np
import pygame


class Pipe:
    def __init__(self, width=50, spacing=200):
        self.screen = pygame.display.get_surface()
        self.spacing = spacing
        self.top = np.random.uniform(
            self.screen.get_size()[1] / 5, (3/4) * self.screen.get_size()[1])
        self.bottom = self.screen.get_size()[1] - (self.top + self.spacing)
        self.x = self.screen.get_size()[0]
        self.width = width
        self.speed = 3.5
        self.rect_top = pygame.Rect(self.x, 0, self.width, self.top)
        self.rect_bot = pygame.Rect(
            self.x, self.top + self.spacing, self.width, self.bottom)
        self.color = (0, 255, 0)

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect_top)
        pygame.draw.rect(self.screen, self.color, self.rect_bot)

    def update(self):
        self.x -= self.speed

        self.rect_top = pygame.Rect(self.x, 0, self.width, self.top)
        self.rect_bot = pygame.Rect(
            self.x, self.top + self.spacing, self.width, self.bottom)

        self.draw()

    def offscreen(self):
        return self.x < -self.width

    @staticmethod
    def get_closest_pipe(pipes, bird_pos_x):
        dist = np.inf
        closest = None

        for pipe in pipes:
            pipe_dist = pipe.x - bird_pos_x
            if 0 < pipe_dist < dist:
                dist = pipe_dist
                closest = pipe

        return closest
