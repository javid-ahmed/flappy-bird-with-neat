import sys
import pygame
from pygame.locals import *
import numpy as np

from genetic_algorithm import Population

from bird import Bird
from pipe import Pipe

# TODO: Enforce max speed for pipe and spawnrate


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
        pygame.display.set_caption("Flappy Bird using Neuroevolution")

        self.pipes = []
        self.pipe_start_spawnrate = 80.9
        self.pipe_current_spawnrate = self.pipe_start_spawnrate
        self.pipe_min_spawnrate = 40
        self.pipe_acc_spawnrate = 0.1
        self.pipe_start_speed = 3.5
        self.pipe_current_speed = self.pipe_start_speed
        self.pipe_max_speed = 10
        self.pipe_acc_speed = 0.03

    def create_population(self, population_size, x, y, width=40, height=40):
        self.birds = []
        self.bird_pos_x = x
        self.bird_pos_y = y
        self.bird_width = width
        self.bird_height = height
        for _ in range(population_size):
            self.birds.append(
                Bird(self.bird_pos_x, self.bird_pos_y, self.bird_width, self.bird_height))

        self.population = Population(self.birds, 0.06)

    def write_text(self, text, x, y):
        text_to_write = self.FONT.render(text, False, (255, 255, 255))
        self.display_surf.blit(text_to_write, (x, y))

    def display_stats(self):
        self.write_text(f"Generation: {self.population.generation}", 0, 0)
        self.write_text(f"Birds alive: {self.population.num_alive}", 0, 20)
        try:
            self.write_text(
                f"Score: {self.population.best_member.score}", 0, 40)
        except AttributeError:
            self.write_text(
                f"Score: 0", 0, 40)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # Game logic goes here
            self.display_surf.fill((0, 0, 0))

            if self.population.num_alive == 0:
                self.population.evaluate()
                self.pipes = []
                self.pipe_current_speed = self.pipe_start_speed
                self.pipe_current_spawnrate = self.pipe_start_spawnrate

            if self.count % int(self.pipe_current_spawnrate) == 0:
                self.pipes.append(
                    Pipe(spacing=220, speed=self.pipe_current_speed))
                self.pipe_current_speed = min(
                    self.pipe_current_speed + self.pipe_acc_speed, self.pipe_max_speed)
                self.pipe_current_spawnrate = max(
                    self.pipe_current_spawnrate - self.pipe_acc_spawnrate, self.pipe_min_spawnrate)
                self.count = 1

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
