import sys
import pygame
from pygame.locals import *
import numpy as np

from genetic_algorithm import Population

from bird import Bird
from pipe import Pipe


class App:
    pygame.init()
    FONT = pygame.font.SysFont("freesansbold.ttf", 24)
    FPS = 60
    FramePerSec = pygame.time.Clock()

    def __init__(self, width, height):
        self.count = 0

        self.screen_width = width
        self.screen_height = height

        self.display_surf = pygame.display.set_mode(
            (self.screen_width, self.screen_height))
        self.display_surf.fill((0, 0, 0))
        pygame.display.set_caption("Flappy Bird using NEAT")

        self.pipes = []
        self.pipe_spawnrate = 60

    def create_population(self, population_size, x, y, width=40, height=40):
        self.birds = []
        self.bird_pos_x = x
        self.bird_pos_y = y
        self.bird_width = width
        self.bird_height = height
        for _ in range(population_size):
            self.birds.append(
                Bird(self.bird_pos_x, self.bird_pos_y, self.bird_width, self.bird_height))

        self.population = Population(self.birds)

    def display_stats(self):
        generation_text = self.FONT.render(
            f"Generation: {self.population.generation}", False, (255, 255, 255))
        self.display_surf.blit(generation_text, (0, 0))

        birds_alive_text = self.FONT.render(
            f"Birds alive: {self.population.num_alive}", False, (255, 255, 255))
        self.display_surf.blit(birds_alive_text, (0, 20))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_SPACE:
                        self.birds[0].jump()
                    if event.key == K_t:
                        self.population.update()

            # Game logic goes here
            self.display_surf.fill((0, 0, 0))

            if self.population.num_alive == 0:
                self.population.evaluate()
                self.pipes = []

            if self.count % self.pipe_spawnrate == 0:
                self.pipes.append(Pipe())

            for pipe in self.pipes:
                if pipe.offscreen():
                    self.pipes.remove(pipe)
                else:
                    pipe.update()

            for bird in self.population.population:
                bird.update(self.pipes)

            self.display_stats()
            pygame.display.update()
            self.FramePerSec.tick(self.FPS)
            self.count += 1


if __name__ == "__main__":
    app = App(600, 800)
    app.create_population(200, 30, 400)
    app.run()
