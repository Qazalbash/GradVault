import pygame as py
from pygame import gfxdraw
from random import *
import pygame_gui
import sys


# main particle class
class Particle():

    def __init__(self, particle_size, width, height, gradient) -> None:
        self.particle_size = particle_size  # to be taken as input
        # if color == 'Red':
        #     self.colors = ['black','red','orange','yellow','white'] # to be taken as input
        # else:
        #     self.colors = ['black','blue','turquoise','white'] # to be taken as input

        self.colors = ['black', 'red', 'orange', 'yellow',
                       'white']  # to be taken as input

        self.gradient = gradient  # variable for transitioning to next color, for smooth transitioning # to be taken as input
        self.colorlist = self.create_colorlist(
        )  # getting our colorlist that now contains all gradients according to our gradient difference set

        # getting how many pixels will fill so that we create the width and height of our particles accordingly
        self.particles_width = width // self.particle_size
        self.particles_height = height // self.particle_size

        self.particles = self.create_particles(
        )  # getting our 2d list of the particles that we want to display

        pass

    # create a color list, that has the different gradeint colours for our particles
    def create_colorlist(self):
        # initialsing our black color, using RGBA Format
        colorlist = [(0, 0, 0, 0)]

        # looping over the colors
        for c, color in enumerate(self.colors[:-1]):
            # using py.Color, that allows RGBA, allowing for opacity that enhances our transitioning
            color1 = py.Color(color)  # assiging our first colour
            color2 = py.Color(
                self.colors[c + 1])  # assigning our color next to that color
            for gap in range(
                    self.gradient
            ):  # loop till the amount of gradient we want, greater gradeint = smoother colors = slower processing
                new_color = color1.lerp(
                    color2, (gap + 0.5) / self.gradient
                )  # interpolating between two colors, creates new color by using lerp function
                colorlist.append(
                    new_color
                )  # adding this to our color list, gradeint being gradually created
        return colorlist
        pass

    # function that creates the base layer of our particles
    def create_particles(self):
        # using a 2d-list to store our particles
        particles = []
        # initialsing our array for our particles based on the dimensions for the particles that we have set
        for c in range(self.particles_height):
            row_particles = []
            for i in range(self.particles_width):
                row_particles.append(0)
            particles.append(row_particles)
        # we will add our base layer of to our particles, this will be our brightest color
        # this will be our bottom row
        bottom_row = self.particles_height - 1
        # position of brightest color
        brightest_idx = len(self.colorlist) - 1
        # looping over last row and adding our brightest color
        for width_idx in range(self.particles_width):
            # takes the position value of brightest color from color list and assigns it to bottom most row
            particles[bottom_row][width_idx] = brightest_idx
        return particles

    # function that adds up the fire particles and asssigns them their color
    def particles_effect(self):
        # looping over columns
        for col in range(self.particles_width):
            # looping over our rows or fire height, except for the frist one as that is black
            for row in range(1, self.particles_height):
                color_index = self.particles[row][col]
                # checking if their is color index already present
                if color_index:
                    # generate a random number, which helps us to create turbulance and a varying color graident
                    vary_gradient = randint(0, 3)
                    # we will be using this formula to shift columns in such a wasy that it does not exceed our fire width and at the
                    #   same time give us a turbulance effect
                    turbulance = col - vary_gradient + 1
                    # asssign the color index values from our color list and mkaing sure that they are not out of range
                    self.particles[row - 1][
                        turbulance %
                        self.particles_width] = color_index - vary_gradient % 2
                else:
                    # if they do not have a color index then those particles lifespan ends, by assigng them the color black, which is the 0 index
                    self.particles[row - 1][col] = 0
        pass

    # function that draws our partices
    def draw(self, display):
        display.fill('black')
        # checking if base layer has been added
        for y, row in enumerate(self.particles):
            for x, color_index in enumerate(row):
                if color_index:
                    # the color is assigned according to the postion of it has in our color list
                    color = self.colorlist[color_index]
                    # drwaing our particles
                    #gfxdraw.circle(display,x*self.particle_size,y*self.particle_size,2,color)
                    #gfxdraw.pixel(display,x*self.particle_size,y*self.particle_size,color)
                    gfxdraw.box(display,
                                (x * self.particle_size, y * self.particle_size,
                                 self.particle_size, self.particle_size), color)
        py.display.flip()
        pass


def main():
    py.init()
    clock = py.time.Clock()
    fps = 60

    # initializing our display
    screen_info = py.display.Info()  # getting screen size
    screen_size = ((screen_info.current_w - 26, screen_info.current_h - 60)
                  )  # adding the screen size to our variable
    width = screen_size[0]
    height = screen_size[1]
    window = (screen_size)  # setting the window to our screen size
    display = py.display.set_mode(window)
    py.display.set_caption("Fire Particle System")

    # create clock object to control framerate
    clock = py.time.Clock()
    fps = 60

    # create UI manager
    ui_manager = pygame_gui.UIManager((width, height))
    # create UI manager
    ui_manager = pygame_gui.UIManager((width, height))

    # create start button
    start_button = pygame_gui.elements.UIButton(relative_rect=py.Rect(
        (width / 2 - 50, height / 2 - 25), (100, 50)),
                                                text="Start",
                                                manager=ui_manager)

    # create gradient input box
    gradient_input = pygame_gui.elements.UITextEntryLine(relative_rect=py.Rect(
        (width / 2 - 50, height / 2 - 90), (100, 30)),
                                                         manager=ui_manager)
    gradient_input.set_allowed_characters(
        "123456789")  # allow only numbers to be entered
    # create label to display prompt to user
    gradient_prompt = pygame_gui.elements.UILabel(relative_rect=py.Rect(
        (width / 2 - 100, height / 2 - 120), (200, 30)),
                                                  text="Enter fire gradient",
                                                  manager=ui_manager)

    # create pixel size input box
    pixel_size_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=py.Rect((width / 2 - 50, height / 2 + 50), (100, 30)),
        manager=ui_manager)
    pixel_size_input.set_allowed_characters(
        "23456789")  # allow only numbers to be entered
    # create label to display prompt to user
    pixel_size_prompt = pygame_gui.elements.UILabel(relative_rect=py.Rect(
        (width / 2 - 100, height / 2 + 90), (200, 30)),
                                                    text="Enter particle size",
                                                    manager=ui_manager)

    # # create list of dropdown options
    # options = ["Red", "Blue"]

    # # create dropdown input box
    # dropdown_input = pygame_gui.elements.UIDropDownMenu(
    #     options_list=options,
    #     starting_option=options[0],
    #     relative_rect=py.Rect((width/2-50, height/2-100), (100, 30)),
    #     manager=ui_manager
    # )

    # # create label to display prompt to user
    # dropdown_prompt = pygame_gui.elements.UILabel(
    #     relative_rect=py.Rect((width/2-100, height/2-130), (200, 30)),
    #     text="Select fire color",
    #     manager=ui_manager
    # )

    # create particle object with default values
    particle = Particle(3, width, height, 7)
    time_delta = 0
    # program loop
    program_running = False
    while True:
        # event loop
        for event in py.event.get():
            if event.type == py.QUIT:
                # quit program when user clicks on close button
                py.quit()
                sys.exit()
            elif event.type == py.USEREVENT:  # check for button press event
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == start_button and event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                        # get values from input boxes
                        gradient = int(gradient_input.get_text())
                        pixel_size = int(pixel_size_input.get_text())
                        # if event.ui_element == dropdown_input:
                        #         # get the selected option
                        #         color = dropdown_input.selected_option

                        # create Particle object with user-selected values
                        particle = Particle(pixel_size, width, height, gradient)

                    # start program loop
                    program_running = True
            # pass events to UI manager
            ui_manager.process_events(event)

        # update UI
        ui_manager.update(time_delta)

        # draw UI
        display.fill('white')
        ui_manager.draw_ui(display)

        # if program is running, update and draw particle
        if program_running:
            particle.particles_effect()
            particle.draw(display)

        # update display
        py.display.update()
        clock.tick(fps)


main()