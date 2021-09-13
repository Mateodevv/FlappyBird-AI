import os
import random

import neat
import pygame

pygame.init()

SCREEN_HEIGHT = 800
SCREEN_WIDTH = 800
BG = pygame.image.load("bg.png")
BIRDIMG = 1
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PIPEGAP = 150
gen = -1

myfont = pygame.font.SysFont("monospace", 15)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()


class Bird:
    def __init__(self, x, y, vel=5, radius=10):
        self.x = x
        self.y = y
        self.vel = vel
        self.radius = radius
        self.isJump = False
        self.jumpCount = 10


class Pipe:
    def __init__(self, x, y, height, width=50, xvel=5):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.xvel = xvel


def jump(bird):
    bird.isJump = True
    bird.jumpCount = 10


def get_pipe(height):
    higherpipe = Pipe(SCREEN_WIDTH, 0, height)
    lowerpipe = Pipe(SCREEN_WIDTH, height + PIPEGAP, height)

    return higherpipe, lowerpipe


def get_rects(pipeone, pipetwo):
    heightlower = pipetwo.y + (800 - pipetwo.y)
    rectupper = pygame.rect.Rect(pipeone.x, pipeone.y, pipeone.width,
                                 pipeone.height)
    rectlower = pygame.rect.Rect(pipetwo.x, pipetwo.y,
                                 pipetwo.width,
                                 heightlower)
    return rectupper, rectlower


def get_collision(birdcollide, pipe1, pipe2):
    if birdcollide.x + 5 >= pipe1.x and birdcollide.x + 5 <= pipe1.x + pipe1.width and birdcollide.y + 5 <= pipe1.y + pipe1.height:
        return True
    elif birdcollide.x + 5 >= pipe2.x and birdcollide.x + 5 <= pipe2.x + pipe2.width and birdcollide.y + 5 >= pipe2.y:
        return True
    elif birdcollide.y < 0:
        return True
    elif birdcollide.y > 795:
        return True
    else:
        return False


def main(genomes, config):
    nets = []
    ge = []
    birds = []

    global gen
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        ge.append(genome)
    pipeList = get_pipe(200)
    score = -1

    gen += 1
    scoredCycle = False

    running = True

    label = myfont.render("Score: ", score, (255, 255, 0))
    labelgen = myfont.render("Generation: " + str(gen), False, (255, 255, 0))

    while running:
        birdslen = len(birds)
        labelbirds = myfont.render("Birds alive: " + str(birdslen), False, (255, 255, 0))
        screen.blit(BG, [0, 0])
        screen.blit(label, (10, 20))
        screen.blit(labelgen, (600, 20))
        screen.blit(labelbirds, (300, 20))
        for birdy in birds:
            pygame.draw.circle(screen, RED, (birdy.x, birdy.y), birdy.radius)
        rect1, rect2 = get_rects(pipeList[0], pipeList[1])
        pygame.draw.rect(screen, (58, 95, 205), rect1)
        pygame.draw.rect(screen, (58, 95, 205), rect2)
        pygame.display.flip()
        clock.tick(60)
        label = myfont.render("Score: " + str(score), 1, (255, 255, 0))
        randomint = random.randint(0, 600)

        for x, birdy in enumerate(birds):
            birdy.y += birdy.vel
            ge[x].fitness += 0.1

            output = nets[x].activate((birdy.y, abs(pipeList[0].height - birdy.y), abs(birdy.y - pipeList[1].y)))

            if output[0] > 0.5:
                jump(birdy)

        if len(birds) < 1:
            break
        if not scoredCycle:
            score += 1
            scoredCycle = True
        for x, birdloop in enumerate(birds):
            if get_collision(birdloop, pipeList[0], pipeList[1]):
                ge[x].fitness -= -1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        pipeList[0].x -= pipeList[0].xvel
        pipeList[1].x -= pipeList[1].xvel
        for birdloop in birds:
            birdloop.y += birdloop.vel
        if pipeList[0].x < - pipeList[0].width:
            for g in ge:
                g.fitness += 5
            pipeList = get_pipe(randomint)
            scoredCycle = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

        for birdloop in birds:
            if birdloop.isJump:
                if birdloop.jumpCount >= -10:

                    birdloop.y -= (birdloop.jumpCount * abs(birdloop.jumpCount)) * 0.2
                    birdloop.y = int(birdloop.y)
                    birdloop.jumpCount -= 1
                else:
                    birdloop.jumpCount = 10
                    birdloop.isJump = False


def run(configpath):
    neatconfig = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, configpath)
    population = neat.Population(neatconfig)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
