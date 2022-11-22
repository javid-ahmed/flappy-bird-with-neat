import pygame
import numpy as np
from pipe import Pipe
from neural_network import NeuralNetwork


class Bird:
    GRAV = 0.8
    LIFT = -16

    def __init__(self, x, y, width, height):
        self.screen = pygame.display.get_surface()
        self.width = width
        self.height = height
        self.x = x
        self.start_y = y
        self.y = y
        self.velocity = 0
        self.min_velocity = -10
        self.alive = True
        self.count = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.color = (np.random.randint(0, 256), np.random.randint(
            0, 256), np.random.randint(0, 256))

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

        self.nn = NeuralNetwork(5, 64, 2)

    def reset(self):
        self.y = self.start_y
        self.velocity = 0
        self.alive = True
        self.count = 0

    def kill(self):
        self.alive = False

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect)

    def update(self, pipes):
        if self.alive:
            self.count += 1

            nearest_pipe = Pipe.get_closest_pipe(pipes, self.x)
            if nearest_pipe == None:
                output = self.nn.feedforward(
                    [
                        self.y / self.screen_height,
                        self.velocity / self.min_velocity,
                        0,
                        0,
                        0
                    ])
            else:
                if self.rect.colliderect(nearest_pipe.rect_top) or self.rect.colliderect(nearest_pipe.rect_bot):
                    self.kill()
                    return
                else:
                    output = self.nn.feedforward(
                        [
                            self.y / self.screen_height,
                            self.velocity / self.min_velocity,
                            nearest_pipe.top / self.screen_height,
                            nearest_pipe.bottom / self.screen_width,
                            nearest_pipe.x / self.screen_width
                        ])

            if output[0] > output[1]:
                self.jump()

            self.velocity += self.GRAV
            self.y += self.velocity

            if self.y + self.height > self.screen_height:
                self.y = self.screen_height - self.height
                self.velocity = 0

            if self.y < 0:
                self.y = 0
                self.velocity = 0

            if self.velocity <= self.min_velocity:
                self.velocity = self.min_velocity

            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
            self.draw()

    def jump(self):
        self.velocity += self.LIFT

    def reset(self):
        self.y = self.start_y
        self.velocity = 0
        self.count = 0

    @property
    def fitness(self):
        return (self.count)**2