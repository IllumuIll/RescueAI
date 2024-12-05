import math
import random
import arcade

from typing import List
from arcade.pymunk_physics_engine import PymunkPhysicsEngine
from arcade import SpriteList, Sprite
from auxilary import Rescuer, Action, Move, Resource

'''
Code skeleton from Python Arcade:
https://api.arcade.academy/en/latest/
'''


class Game(arcade.Window):
    """ Main Game """

    def __init__(self, width, height, title):
        """ Init """
        super().__init__(width, height, title, visible=True)

        self.background = arcade.load_texture(
            "textures/background.png")

        self.rescuer_list: SpriteList
        self.mother_ship_list: SpriteList
        self.alien_list: SpriteList
        self.resource_list: SpriteList
        self.asteroids_list: SpriteList
        self.wall_list: SpriteList

        self.physics_engine: PymunkPhysicsEngine

        self.action_list: List[Action] = []

        self.width = width
        self.height = height

        self._sprite_scaling = 0.5
        self._sprite_image_size = 128
        self._sprite_size = int(self._sprite_scaling * self._sprite_image_size)

        self.frames = []
        self.num = 0

    def setup(self):

        self.rescuer_list = SpriteList(use_spatial_hash=True)
        self.mother_ship_list = SpriteList(use_spatial_hash=True)
        self.alien_list = SpriteList()
        self.resource_list = SpriteList()
        self.asteroids_list = SpriteList(use_spatial_hash=True)
        self.wall_list = SpriteList(use_spatial_hash=True)

        self.pick_up = 0
        self.delivery = 0
        self.collision = 0

    def reset(self):
        self.rescuer_list.clear()
        self.mother_ship_list.clear()
        self.alien_list.clear()
        self.resource_list.clear()
        self.asteroids_list.clear()
        self.wall_list.clear()

        # Create first rescuer
        rescuer: Sprite = Rescuer(
            filename="textures/rescuer.png",
            scale=self._sprite_scaling,
            center_x=self._get_random_coord(
                lb=100),
            center_y=self._get_random_coord(
                lb=100),
            health=5)
        self.rescuer_list.append(rescuer)

        mothership_x, mothership_y, alien_x, alien_y = self._no_overlapping_coords()

        # Provide the mothership
        mothership: Sprite = Sprite(
            filename="textures/mothership.png",
            scale=self._sprite_scaling / 2,
            center_x=mothership_x,
            center_y=mothership_y)

        self.mother_ship_list.append(mothership)

        # Provide the alien, Alfred
        alien: Sprite = Sprite(
            filename="textures/alfred.png",
            scale=self._sprite_scaling / 2.5,
            center_x=alien_x,
            center_y=alien_y)

        self.alien_list.append(alien)

        for _ in range(8):
            # Provide the the astroids
            texture_name = self._get_random_astroid_texture()
            start_x, start_y = self._get_random_astroid_coord()
            scaling = random.randint(1, 3)
            astroid: Sprite = Sprite(
                filename=texture_name,
                scale=self._sprite_scaling / 1.5 * scaling,
                center_x=start_x,
                center_y=start_y)

            self.asteroids_list.append(astroid)

        # Set up the walls
        for x in range(0, self.width + 1, self._sprite_size):
            wall = arcade.Sprite(":resources:images/tiles/brickGrey.png",
                                 self._sprite_scaling)
            wall.center_x = x
            wall.center_y = 0
            self.wall_list.append(wall)

            wall = arcade.Sprite(":resources:images/tiles/brickGrey.png",
                                 self._sprite_scaling)
            wall.center_x = x
            wall.center_y = self.height
            self.wall_list.append(wall)

        # Set up the walls
        for y in range(self._sprite_size, self.height, self._sprite_size):
            wall = arcade.Sprite(":resources:images/tiles/brickGrey.png",
                                 self._sprite_scaling)
            wall.center_x = 0
            wall.center_y = y
            self.wall_list.append(wall)

            wall = arcade.Sprite(":resources:images/tiles/brickGrey.png",
                                 self._sprite_scaling)
            wall.center_x = self.width
            wall.center_y = y
            self.wall_list.append(wall)

        # ---------------------------------------------------------------------------------------------------
        # --------------------------------Pymunk Physics Engine Setup ---------
        # ---------------------------------------------------------------------------------------------------
        damping = 1
        gravity = (0, 0)

        self.physics_engine = PymunkPhysicsEngine(damping=damping,
                                                  gravity=gravity)

        # Add rescuer to physics engine
        for rescuer in self.rescuer_list:
            self.physics_engine.add_sprite(
                rescuer,
                friction=0.2,
                moment_of_inertia=PymunkPhysicsEngine.MOMENT_INF,
                damping=1,
                collision_type="rescuer",
                max_velocity=400)

        # Add mothership to physics engine
        self.physics_engine.add_sprite_list(
            self.mother_ship_list,
            friction=0.2,
            collision_type="mothership",
            body_type=PymunkPhysicsEngine.STATIC)

        # Add alien to physics engine
        for alien in self.alien_list:
            self.physics_engine.add_sprite(
                alien,
                friction=0.0,
                moment_of_inertia=PymunkPhysicsEngine.DYNAMIC,
                damping=0.9,
                collision_type="alien",
                max_velocity=400)

        # Add alien to physics engine
        for astroid in self.asteroids_list:
            self.physics_engine.add_sprite(
                astroid,
                friction=0.0,
                moment_of_inertia=PymunkPhysicsEngine.DYNAMIC,
                damping=1,
                collision_type="astroid",
                max_velocity=400)

        for astroid in self.asteroids_list:
            self.physics_engine.apply_force(
                astroid, self._get_random_force(astroid))

        # Add walls to physics engine
        self.physics_engine.add_sprite_list(
            self.wall_list,
            friction=0.0,
            collision_type="wall",
            body_type=PymunkPhysicsEngine.STATIC)

        def rescuer_alien_collision_handler(
                sprite_a, sprite_b, arbiter, space, data):
            """ Called for rescuer/alien collision """

            # Retrieve rescuer considered in the collision
            rescuer = arbiter.shapes[0]
            alien = arbiter.shapes[1]

            rescuer: Rescuer = self.physics_engine.get_sprite_for_shape(
                rescuer)

            alien: Sprite = self.physics_engine.get_sprite_for_shape(
                alien)

            # Check if the rescuer is free to carry a resource
            if not rescuer.carries_resource:
                print('Collecting Resource')
                self.pick_up += 1
                rescuer.carries_resource = True

                # Instantiate new resource to be carried
                resource: Resource = Resource(
                    filename="textures/alfred.png",
                    scale=self._sprite_scaling / 3,
                    center_x=rescuer.center_x,
                    center_y=rescuer.center_y)

                # Stick resource to rescuer
                resource.is_stuck = True
                resource.center_x = rescuer.center_x - 10
                resource.center_y = rescuer.center_y - 10

                # Append resource to list and add it to the physics engine
                self.resource_list.append(resource)

                rescuer.resource_carried = resource
                alien.remove_from_sprite_lists()

        def rescuer_mother_ship_collision_handler(
                sprite_a, sprite_b, arbiter, space, data):
            """ Called for rescuer/mother_shipt collision """

            # Retrieve rescuer considered in the collision
            rescuer: Rescuer = arbiter.shapes[0]
            rescuer = self.physics_engine.get_sprite_for_shape(rescuer)

            # Check if rescuer is carrying resources when it touches the head
            # quarter
            if rescuer.carries_resource:
                print('Delivering Resource')
                self.delivery += 1
                rescuer.carries_resource = False

                # Let resource disappear, as it has been delivered
                resource: Resource = rescuer.resource_carried
                resource.remove_from_sprite_lists()

                rescuer.resource_carried = None

        def ignore_collision(sprite_a, sprite_b, arbiter, space, data):
            return False

        self.physics_engine.add_collision_handler(
            "wall",
            "astroid",
            begin_handler=ignore_collision)

        self.physics_engine.add_collision_handler(
            "rescuer",
            "alien",
            post_handler=rescuer_alien_collision_handler)
        self.physics_engine.add_collision_handler(
            "rescuer", "mothership", post_handler=rescuer_mother_ship_collision_handler)

    # ---------------------------------------------------------------------------------------------------
    # ---------------------------------------- UPDATE HANDLING ---------------
    # ---------------------------------------------------------------------------------------------------

    def custom_draw(self):
        arcade.start_render()
        # arcade.set_background_color(arcade.color.BLACK)
        arcade.draw_lrwh_rectangle_textured(
            0, 0, self.width, self.height, self.background)
        self.rescuer_list.draw()
        self.mother_ship_list.draw()
        self.alien_list.draw()
        self.resource_list.draw()
        self.asteroids_list.draw()

        frame = arcade.get_image()
        self.frames.append(frame)

    def custom_update(self) -> bool:
        # Apply all queued actions
        for action in self.action_list:
            move_action: Move = action
            self.physics_engine.apply_impulse(
                move_action.sprite, move_action.force)

            # Move resource along rescuer
            if isinstance(action.sprite,
                          Rescuer) and action.sprite.carries_resource:
                rescuer: Rescuer = action.sprite
                resource: Resource = rescuer.resource_carried
                resource.center_x, resource.center_y = rescuer.center_x, rescuer.center_y - 20

        # Clear the list after application
        self.action_list.clear()

        # Update the environment via the physics engine
        self.physics_engine.step()

        # Manage all astroids
        to_be_removed_astroids = []
        for astroid in self.asteroids_list:
            if self._has_moved_beyond_screen(astroid):
                # In case we have an astroid wandering off, remove it
                to_be_removed_astroids.append(astroid)
                # Add a new astroid with random force and starting position on
                # the borders
                texture_name = self._get_random_astroid_texture()
                start_x, start_y = self._get_random_astroid_coord()
                astroid: Sprite = Sprite(
                    filename=texture_name,
                    scale=self._sprite_scaling / 1.5,
                    center_x=start_x,
                    center_y=start_y)

                self.physics_engine.add_sprite(
                    astroid,
                    friction=0.0,
                    moment_of_inertia=PymunkPhysicsEngine.STATIC,
                    damping=1,
                    collision_type="astroid",
                    max_velocity=400)

                self.physics_engine.apply_impulse(
                    astroid, self._get_random_force(astroid))

                self.asteroids_list.append(astroid)

        for astroid in to_be_removed_astroids:
            astroid.remove_from_sprite_lists()

        collided_with_astroid = False if len(
            arcade.check_for_collision_with_list(
                self.rescuer_list[0],
                self.asteroids_list)) == 0 else True

        rescuer_in_environment = True if self._has_moved_beyond_screen(
            self.rescuer_list[0]) else False

        if collided_with_astroid:
            print("Collision")
            self.collision += 1

        return collided_with_astroid or rescuer_in_environment

    def get_image(self, x: int, y: int, width: int, height: int):
        return arcade.get_image(x, y, width=width, height=height)

    def _get_random_coord(self, lb) -> tuple:
        lower_bound = lb
        upper_bound = self.height - lb
        return random.randint(lower_bound, upper_bound)

    def _get_random_astroid_coord(self):
        border = random.choice(['top', 'bottom', 'left', 'right'])
        margin = random.randint(15, 50)
        if border == 'top':
            x = random.randint(margin, self.width)
            y = margin
        elif border == 'bottom':
            x = random.randint(margin, self.width)
            y = self.height - margin
        elif border == 'left':
            x = margin
            y = random.randint(margin, self.height)
        elif border == 'right':
            x = self.width - margin
            y = random.randint(margin, self.height)
        return (x, y)

    def _get_random_force(self, asteroid: Sprite) -> tuple:
        center_x = self.width / 2
        center_y = self.height / 2
        if asteroid.center_x < center_x:
            x_force = random.randint(50, 100)
        else:
            x_force = random.randint(-100, -50)

        if asteroid.center_y < center_y:
            y_force = random.randint(50, 100)
        else:
            y_force = random.randint(-100, -50)
        return (x_force, y_force)

    def _get_random_astroid_texture(self):
        return {
            0: "textures/astroid_1.png",
            1: "textures/astroid_1.png"
        }[random.randint(0, 1)]

    def _has_moved_beyond_screen(self, sprite: Sprite) -> bool:
        if sprite.left < 0 or \
                sprite.right > self.width or \
                sprite.bottom < 0 or \
                sprite.top > self.height:
            return True
        else:
            return False

    def _no_overlapping_coords(self) -> tuple:
        # Non-blocking random x,y pair
        for _ in range(100):
            x_1, y_1, x_2, y_2 = self._get_random_coord(
                lb=50), self._get_random_coord(
                lb=50), self._get_random_coord(
                lb=50), self._get_random_coord(
                lb=50)
            if math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2) > 200:
                break
        return x_1, y_1, x_2, y_2
    
    def _save_video(self):
        self.frames[0].save(
            "Rescue-Mission_" + str(self.num) + ".gif",
            save_all=True,
            append_images=self.frames[1:],
            duration=1000 / 45,  # 60 fps
            loop=0
        )
        self.frames.clear()
        self.num += 1
        print("Video saved as Rescue-Mission.gif")

