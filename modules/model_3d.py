import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import cv2


def main(rotation_vector, translation_vector):
    pygame.init()
    display = (500, 500)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glOrtho(-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glPushMatrix()
        glTranslatef(translation_vector[0][0], -translation_vector[1][0], 0)
        glRotatef(57.3 * rotation_vector[0][0], -1, 0, 0)
        glRotatef(57.3 * rotation_vector[1][0], 0, 1, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glColor3f(0, 1, 0)
        glBegin(GL_QUADS)
        glVertex3f(-5, -5, 0)
        glVertex3f(5, -5, 0)
        glVertex3f(5, 5, 0)
        glVertex3f(-5, 5, 0)
        glEnd()

        glColor3f(1, 1, 1)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, -20)
        glEnd()

        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)
