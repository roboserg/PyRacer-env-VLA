import pygame, os

class Car():
    def __init__(self, game):
        self.game = game
        self.position, self.distance = 0, 0
        self.position_int = self.game.DISPLAY_W / 2 + int(self.game.SCREEN_WIDTH * self.position / 2) - 14
        self.speed = 0
        self.curvature = 0

        # Target base height for the car in-game
        self.base_h = 40
        
        # Load the tilemap with transparency
        sheet_path = os.path.join(self.game.img_dir, "cars-tilemap.png")
        self.sheet = pygame.image.load(sheet_path).convert_alpha()

        # Sprite coordinates [x, y, w, h] from our blob analysis
        coords = [
            (157, 12, 262, 207),  # Top (Back View)
            (103, 295, 380, 222), # Mid (Slight Turn)
            (14, 559, 552, 249)   # Bot (Hard Turn)
        ]

        # Store raw surfaces to scale them dynamically during draw()
        def get_raw(c):
            surf = pygame.Surface((c[2], c[3]), pygame.SRCALPHA)
            surf.blit(self.sheet, (0, 0), c)
            return surf

        # Create our raw sprite bank
        self.raw_sprites = {
            "straight": get_raw(coords[0]),
            "slight_right": get_raw(coords[1]),
            "hard_right": get_raw(coords[2])
        }

        # Generate left turns by flipping right turns horizontally
        self.raw_sprites["slight_left"] = pygame.transform.flip(self.raw_sprites["slight_right"], True, False)
        self.raw_sprites["hard_left"] = pygame.transform.flip(self.raw_sprites["hard_right"], True, False)
        
        # Initial image setup (will be updated in draw())
        self.image = self.raw_sprites["straight"]
        
        # Steering timer for hard turn transition
        self.steer_timer = 0
        self.hard_steer_threshold = 2.0 # Seconds before switching to hard turn
        
        # Base vertical position for the car
        self.base_y = 220

    def get_current_scaled_image(self, name, current_h):
        """Helper to scale a sprite based on current target height"""
        raw = self.raw_sprites[name]
        aspect_ratio = raw.get_width() / raw.get_height()
        target_w = int(current_h * aspect_ratio)
        return pygame.transform.scale(raw, (target_w, current_h))

    def clamp_speed(self):
        self.speed = max(0, self.speed)
        self.speed = min(self.speed,1)

    def update(self):
        # Update the Car's movement
        if self.game.actions['accel']:
            self.speed += .5 * self.game.dt
        else:
            self.speed -= .25 * self.game.dt

        if self.game.actions['brake']:
            self.speed -=.75 * self.game.dt

        if self.game.actions['left'] or self.game.actions['right']:
            self.steer_timer += self.game.dt
            if self.game.actions['left']:
                self.curvature -= .3 * self.game.dt
            if self.game.actions['right']:
                self.curvature += .3 * self.game.dt
        else:
            self.steer_timer = 0

        if abs(self.curvature - self.game.map.track_curvature) >= .6:
            self.speed -= 5 * self.game.dt

        self.clamp_speed()
        self.distance += 70 * self.speed * self.game.dt

    def draw(self):
        # Determine current target height based on speed (perspective)
        # Higher speed = higher on screen = smaller sprite
        # 40 (base) -> 32 (max speed - 20% reduction)
        current_h = int(self.base_h * (1.0 - (self.speed * 0.20)))

        # Determine which sprite name to use
        actions = self.game.actions
        sprite_name = "straight"

        if actions.get("left"):
            sprite_name = "hard_left" if self.steer_timer > self.hard_steer_threshold else "slight_left"
        elif actions.get("right"):
            sprite_name = "hard_right" if self.steer_timer > self.hard_steer_threshold else "slight_right"
        
        # Get scaled image for current frame
        self.image = self.get_current_scaled_image(sprite_name, current_h)

        # Track curvature difference is only for horizontal position on screen
        steer_diff = self.curvature - self.game.map.track_curvature
        self.position = steer_diff
        
        # Centering logic: use the sprite's current width to offset the blit position
        img_w = self.image.get_width()
        self.position_int = self.game.DISPLAY_W / 2 + int(self.game.DISPLAY_W * self.position / 2) - (img_w // 2)
        
        # Vertical movement based on speed
        forward_offset = int(self.speed * 44)
        current_y = self.base_y - forward_offset
        
        self.game.display.blit(self.image, (self.position_int, current_y))
