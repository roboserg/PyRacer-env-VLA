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
        self.slight_steer_threshold = 0.25 # Seconds before switching to slight turn
        self.hard_steer_threshold = 1.5 # Seconds before switching to hard turn
        self.last_steer_dir = None # Track 'left', 'right', or None
        
        # Base vertical position for the car
        self.base_y = 220
        self.current_draw_y = 220
        
        # Particle systems
        self.dirt_particles = []

    def get_current_scaled_image(self, name, current_h):
        """Helper to scale a sprite based on current target height"""
        raw = self.raw_sprites[name]
        aspect_ratio = raw.get_width() / raw.get_height()
        target_w = int(current_h * aspect_ratio)
        return pygame.transform.scale(raw, (target_w, current_h))

    def clamp_speed(self):
        self.speed = max(0, self.speed)
        self.speed = min(self.speed, 1.5) # Increased max speed limit to 2.0 (200 Km/h)

    def update(self):
        # Update the Car's movement
        if self.game.actions['accel']:
            self.speed += .5 * self.game.dt
        else:
            self.speed -= .25 * self.game.dt

        if self.game.actions['brake']:
            self.speed -=.75 * self.game.dt

        if self.game.actions['left'] or self.game.actions['right']:
            current_dir = 'left' if self.game.actions['left'] else 'right'
            # Reset timer if we switched directions
            if current_dir != self.last_steer_dir:
                self.steer_timer = 0
            
            self.steer_timer += self.game.dt
            self.last_steer_dir = current_dir
            
            # Increased steering sensitivity for sharper turns (0.3 -> 0.45)
            if self.game.actions['left']:
                self.curvature -= .4 * self.game.dt
            if self.game.actions['right']:
                self.curvature += .4 * self.game.dt
        else:
            self.steer_timer = 0
            self.last_steer_dir = None

        # Collision with road wall / off-road
        steer_diff = self.curvature - self.game.map.track_curvature
        
        # Softened road boundaries with a hard screen limit
        if steer_diff > 0.6:
            self.curvature -= 0.5 * self.game.dt # Increased push back
            self.speed -= 1.5 * self.game.dt # Speed penalty
            if self.speed > 0.05: self.spawn_dirt(0.6)
            # Hard physical limit to ensure car stays on screen grass
            if steer_diff > 0.8: self.curvature = self.game.map.track_curvature + 0.8
        elif steer_diff < -0.6:
            self.curvature += 0.5 * self.game.dt # Increased push back
            self.speed -= 1.5 * self.game.dt # Speed penalty
            if self.speed > 0.05: self.spawn_dirt(-0.6)
            # Hard physical limit to ensure car stays on screen grass
            if steer_diff < -0.8: self.curvature = self.game.map.track_curvature - 0.8
        
        self.clamp_speed()
        self.distance += 70 * self.speed * self.game.dt
        
        # Update particles
        self.update_dirt()

    def spawn_dirt(self, steer_diff):
        import random
        # Spawn dirt near the wheels using SCREEN coordinates for perfect alignment
        for _ in range(3):
            # Center dirt on the sprite's width
            img_w = self.image.get_width()
            img_h = self.image.get_height()
            
            p = {
                "x": self.position_int + (img_w // 2) + random.randint(-15, 15),
                "y": self.current_draw_y + img_h - 5,
                "vx": random.uniform(-1.0, 1.0), 
                "vy": random.uniform(1.0, 3.0), # Move "down" the screen
                "life": 1.0,
                "size": random.randint(3, 6),
                "color": (random.randint(100, 150), random.randint(70, 100), 20)
            }
            self.dirt_particles.append(p)

    def update_dirt(self):
        for p in self.dirt_particles[:]:
            # Screen-space physics: move down and slightly horizontally
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            # Slower life decay
            p["life"] -= self.game.dt * 0.8
            if p["life"] <= 0:
                self.dirt_particles.remove(p)

    def draw_dirt(self):
        for p in self.dirt_particles:
            alpha = int(p["life"] * 255)
            # Scaling size slightly as it flows back
            size = int(p["size"] * (1.0 + (1.0 - p["life"]) * 0.5))
            
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*p["color"], alpha))
            self.game.display.blit(s, (p["x"], p["y"]))

    def draw(self):
        # Vertical position first — needed to derive perspective depth for sizing
        forward_offset = int(self.speed * 44)
        self.current_draw_y = self.base_y - forward_offset

        # Determine which sprite name to use
        actions = self.game.actions
        if actions.get("left") and self.steer_timer > self.slight_steer_threshold:
            sprite_name = "hard_left" if self.steer_timer > self.hard_steer_threshold else "slight_left"
        elif actions.get("right") and self.steer_timer > self.slight_steer_threshold:
            sprite_name = "hard_right" if self.steer_timer > self.hard_steer_threshold else "slight_right"
        else:
            sprite_name = "straight"

        # Two-pass sizing: car should only shrink once it visually lifts off the
        # screen bottom, not while it's still clamped there by a large sprite.
        mid_h = self.game.map.mid_h
        render_height = self.game.DISPLAY_H - mid_h
        straight = self.raw_sprites["straight"]
        straight_aspect = straight.get_width() / straight.get_height()
        # k converts a road-width fraction (0..1 screen width) to sprite height
        k = self.game.DISPLAY_W * 0.30 / straight_aspect

        raw_draw_y = self.current_draw_y  # = base_y - forward_offset

        # Pass 1: rough height from raw y
        p1 = max(0.05, min(1.0, (raw_draw_y - mid_h) / render_height))
        h1 = max(10, int((0.2 + p1 * 0.7) * k))

        # Clamp with pass-1 height, then recompute perspective from clamped y
        clamped_y = min(raw_draw_y, self.game.DISPLAY_H - h1)
        p2 = max(0.05, min(1.0, (clamped_y - mid_h) / render_height))
        current_h = max(10, int((0.2 + p2 * 0.7) * k))

        # Get scaled image for current frame
        self.image = self.get_current_scaled_image(sprite_name, current_h)

        # Track curvature difference is only for horizontal position on screen
        steer_diff = self.curvature - self.game.map.track_curvature
        self.position = steer_diff

        # Centering logic
        img_w = self.image.get_width()
        img_h = self.image.get_height()
        self.position_int = self.game.DISPLAY_W / 2 + int(self.game.DISPLAY_W * self.position / 2) - (img_w // 2)

        # Clamp both axes so car never disappears off screen
        self.position_int = max(0, min(self.position_int, self.game.DISPLAY_W - img_w))
        self.current_draw_y = max(0, min(raw_draw_y, self.game.DISPLAY_H - img_h))
        
        # Draw dirt BEFORE car
        self.draw_dirt()
        
        self.game.display.blit(self.image, (self.position_int, self.current_draw_y))
