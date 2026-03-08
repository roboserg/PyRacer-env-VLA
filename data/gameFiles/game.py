import pygame, time, os
from data.gameFiles.map import Map
from data.util.controls import load_controls
from data.util.fps import FPS
from data.menus.main_menu import MainMenu
from data.menus.credits_menu import CreditsMenu
from data.menus.options_menu import OptionsMenu
from data.menus.controls_menu import ControlsMenu


class Game:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 1, 512)  # Prevents delay in jumping sound
        pygame.init()
        # Intitialize display surface and screen
        self.TITLE = "Racing Game Demo"
        self.DISPLAY_W, self.DISPLAY_H = (
            480,
            270,
        )  # Dimensions of the 'canvas'. Use this for game logic May be upscaled later
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = (
            480 * 2,
            270 * 2,
        )  # Dimensions of the screen that canvas will be drawn onto
        self.display = pygame.Surface((self.DISPLAY_W, self.DISPLAY_H))
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.set_alpha(None)
        # Initialize clock, controls, and other variables
        self.running, self.playing = True, False
        self.load_directories()
        self.font_cache = {}  # Cache fonts by size to avoid recreating them every frame
        self.speed_history = []  # Track speed for averaging
        self.clock = FPS(
            60
        )  # Argument in FPS class will cap the FPS at the given value
        self.load_controls()
        self.load_menus()
        self.load_images()

        # Resets any logic that needs to be reset after every game loop
        # May include player, enemies, level, objects, etc

    def reset(self):
        self.map = Map(self)
        self.go_text = 0
        self.lap_time = 0
        self.countdown = 3
        self.countdownUpdate = time.time()
        self.counting_down = True
        self.complete = False
        self.finished_countdown = 0
        self.speed_history = []  # Reset speed history for new race
        self.light_sound.play()

    # Main Game Loop. Starts by resetting the game, and then loops until playing is set to false
    def game_loop(self):
        self.reset()
        pygame.mixer.music.play(-1)
        while self.playing:
            self.get_dt()
            self.get_events()
            if self.countdown > 0:
                self.count_down()
            else:
                self.update()
            self.render()
        pygame.mixer.music.stop()

    def get_events(self):
        # Gets all events from the user, stores them in the 'actions' dictionary
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.playing = False
                self.running = False
                self.current_menu.run_display = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.playing = False
                    self.running = False
                    self.current_menu.run_display = False
                if event.key == self.controls["Left"]:
                    self.actions["left"] = True
                if event.key == self.controls["Right"]:
                    self.actions["right"] = True
                if event.key == self.controls["Up"]:
                    self.actions["accel"] = True
                if event.key == self.controls["Down"]:
                    self.actions["brake"] = True
                if event.key == self.controls["Start"]:
                    self.actions["start"] = True
                if event.key == self.controls["Run"]:
                    self.actions["run"] = True
                if event.key == pygame.K_r:
                    self.randomize_car_state()

            if event.type == pygame.KEYUP:
                if event.key == self.controls["Left"]:
                    self.actions["left"] = False
                if event.key == self.controls["Right"]:
                    self.actions["right"] = False
                if event.key == self.controls["Up"]:
                    self.actions["accel"] = False
                if event.key == self.controls["Down"]:
                    self.actions["brake"] = False
                if event.key == self.controls["Start"]:
                    self.actions["start"] = False
                if event.key == self.controls["Run"]:
                    self.actions["run"] = False

    def randomize_car_state(self):
        """Randomly places the car on the track with random speed for testing."""
        import random
        if hasattr(self, "map") and hasattr(self.map, "car"):
            car = self.map.car
            # Random distance along the track
            car.distance = random.uniform(0, self.map.track_length)
            # Random lateral position (within road bounds)
            # Use self.map.curvature as baseline for "on-track"
            car.curvature = self.map.track_curvature + random.uniform(-0.5, 0.5)
            # Random speed (0.2 to 1.5)
            car.speed = random.uniform(0.2, 1.5)
            print(f"DEBUG: Randomized car to dist={car.distance:.1f}, speed={car.speed:.2f}")

    # Update any of the game sprites
    def update(self):
        self.timer()
        self.map.update()
        if self.complete:
            self.complete_timer()

    # Draw all sprites and images onto the screen
    def render(self):
        self.map.render()
        self.draw_startup()
        self.draw_screen()

    # Helper function for render(). Draws the canvas onto the actual game window
    def draw_screen(self):
        self.screen.blit(
            pygame.transform.scale(
                self.display, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            ),
            (0, 0),
        )
        pygame.display.update()

    def draw_startup(self):
        if self.countdown > 0:
            self.display.blit(
                self.light_images[self.countdown - 1], (self.DISPLAY_W * 0.5 - 32, 50)
            )
        elif self.go_text < 0.75:
            self.go_sound.play()
            self.go_text += self.dt
            self.display.blit(self.go_img, (self.DISPLAY_W * 0.5 - 32, 50))

    # Computes FPS, delta time, and caps the frame rate if specified in the FPS() class
    def get_dt(self):
        elapsed_time = self.clock.elapsed_time()
        self.dt = elapsed_time
        pygame.display.set_caption(
            "{0}: {1:.2f}".format(self.TITLE, self.clock.get_fps())
        )

    # Loads all menu classes
    def load_menus(self):
        self.main_menu = MainMenu(self)
        self.credits_menu = CreditsMenu(self)
        self.options_menu = OptionsMenu(self)
        self.controls_menu = ControlsMenu(self)
        self.current_menu = self.main_menu
        self.font = pygame.font.Font(self.main_menu.font_name, 16)

    def load_controls(self):
        self.controls = load_controls(
            self.dir
        )  # Loads saved controls from the included .json file
        self.actions = {
            "left": False,
            "right": False,
            "accel": False,
            "brake": False,
            "jump": False,
            "start": False,
            "run": False,
        }

        # Resets all the keys to False. Useful for Menus

    def reset_keys(self):
        for key in self.actions:
            self.actions[key] = False

    def draw_text(self, text, size, color, x, y):
        # Use cached font to avoid creating new font objects every frame
        if size not in self.font_cache:
            self.font_cache[size] = pygame.font.Font(self.font_name, size)
        font = self.font_cache[size]
        text_surface = font.render(text, True, color)
        text_surface.set_colorkey((0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.x, text_rect.y = x, y
        self.display.blit(text_surface, text_rect)

    def timer(self):
        self.lap_time += self.dt

        # Track speed for averaging
        if self.map:
            self.speed_history.append(self.map.car.speed)
            # Keep only last 3600 entries (60 seconds at 60 FPS)
            if len(self.speed_history) > 3600:
                self.speed_history.pop(0)

    def get_average_speed(self):
        """Get average speed since race started"""
        if not self.speed_history:
            return 0
        return sum(self.speed_history) / len(self.speed_history)

    def get_average_speed_60s(self):
        """Get average speed for last 60 seconds"""
        if not self.speed_history:
            return 0
        last_60_seconds = (
            self.speed_history[-3600:]
            if len(self.speed_history) >= 3600
            else self.speed_history
        )
        if not last_60_seconds:
            return 0
        return sum(last_60_seconds) / len(last_60_seconds)

    def complete_timer(self):
        self.finished_countdown += self.dt
        if self.finished_countdown > 3:
            self.playing = False

    def count_down(self):
        now = time.time()
        if now - self.countdownUpdate > 1:
            self.countdownUpdate = now
            self.countdown -= 1
            self.light_sound.play()
        if self.countdown > 0:
            self.counting_down = False

    def load_directories(self):
        self.dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )  # Goes from gameFiles to data
        self.img_dir = os.path.join(self.dir, "images")
        self.sound_dir = os.path.join(self.dir, "sounds")
        self.font_name = os.path.join(self.dir, "menus", "november.ttf")
        self.theme = pygame.mixer.music.load(
            os.path.join(self.sound_dir, "racing_song.ogg")
        )
        self.light_sound = pygame.mixer.Sound(os.path.join(self.sound_dir, "light.wav"))
        self.go_sound = pygame.mixer.Sound(os.path.join(self.sound_dir, "go.wav"))
        self.light_sound.set_volume(0.3)
        self.go_sound.set_volume(0.2)

    def load_images(self):
        self.light_images = [
            pygame.image.load(os.path.join(self.img_dir, "streetlight3.png")).convert(),
            pygame.image.load(os.path.join(self.img_dir, "streetlight2.png")).convert(),
            pygame.image.load(os.path.join(self.img_dir, "streetlight1.png")).convert(),
        ]
        for image in self.light_images:
            image.set_colorkey((0, 0, 0))
        self.go_img = pygame.image.load(os.path.join(self.img_dir, "GO.png")).convert()
        self.go_img.set_colorkey((0, 0, 0))
