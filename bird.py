import pygame
import numpy as np
from pipe import Pipe
from neural_network import NeuralNetwork


class Bird:
    GRAV = 1
    LIFT = -20

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
        self.score = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.color = (np.random.randint(0, 256), np.random.randint(
            0, 256), np.random.randint(0, 256))

        self.screen_width = self.screen.get_size()[0]
        self.screen_height = self.screen.get_size()[1]

        self.nn = NeuralNetwork(5, 64, 2)

    def reset(self):
        self.y = self.start_y
        self.velocity = 0
        self.score = 0
        self.alive = True

    def kill(self):
        self.alive = False

    def draw(self):
        pygame.draw.rect(self.screen, self.color, self.rect)

    def jump(self):
        self.velocity += self.LIFT

    def update(self, pipes):
        if not self.alive:
            return

        if self.y > self.screen_height - self.height or self.y < 0:
            self.kill()
            return

        inputs = [
            self.y / self.screen_height,
            self.velocity / self.min_velocity,
            0,
            0,
            0
        ]

        nearest_pipe = Pipe.get_closest_pipe(pipes, self.x)
        if nearest_pipe is not None:
            if self.rect.colliderect(nearest_pipe.rect_top) or self.rect.colliderect(nearest_pipe.rect_bot):
                self.kill()
                return
            else:
                inputs[2] = nearest_pipe.top / self.screen_height
                inputs[3] = nearest_pipe.bottom / self.screen_height
                inputs[4] = nearest_pipe.x / self.screen_width

        output = self.nn.feedforward(inputs)

        if output[0] > output[1]:
            self.jump()

        self.velocity += self.GRAV
        self.velocity = max(self.velocity, self.min_velocity)
        self.y += self.velocity

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.draw()
        self.score += 1

    def crossover(self, parentA, parentB, mutation_rate):
        self.nn.crossover(parentA.nn, parentB.nn, mutation_rate)

    def apply(self):
        self.nn.apply()

    @property
    def fitness(self):
        return (self.score)**2
