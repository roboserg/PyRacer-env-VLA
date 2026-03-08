import pygame, math, os, random
from data.gameFiles.car import Car


class Map:
    def __init__(self, game):
        self.game = game
        self.mid_w, self.mid_h = (
            int(self.game.DISPLAY_W / 2),
            int(self.game.DISPLAY_H / 2),
        )
        self.car = Car(self.game)
        self.curvature = 0
        self.track_curvature = 0
        self.track_length = 0
        self.load_track()
        self.lap, self.counted = 0, True
        self.lap_times = []
        self.background_img = pygame.image.load(
            os.path.join(self.game.img_dir, "background.png")
        ).convert()

    def render(self):
        # self.update()
        self.draw_map()

    def update(self):
        # Update the car
        self.car.update()
        self.update_track()
        if self.lap > 2:
            self.lap = 2
            self.game.complete = True

    def update_track(self):
        track_section, offset = 0, 0

        # Generate more track if needed (endless track)
        self.update_track_generation(self.car.distance)

        if self.car.distance >= self.track_length:
            self.car.distance -= self.track_length

        # optimise Curvature Calculations
        while track_section < len(self.track) and offset <= self.car.distance:
            offset += self.track[track_section][1]
            track_section += 1
        target_curvature = self.track[track_section - 1][0]

        track_curve_diff = (
            (target_curvature - self.curvature) * self.game.dt * self.car.speed
        )
        self.curvature += track_curve_diff

        self.track_curvature += self.curvature * self.game.dt * self.car.speed

        if track_section == 1 and not self.counted:
            self.store_times()
            self.counted = True
            self.lap += 1

        if track_section > 1:
            self.counted = False

    def draw_map(self):

        self.game.display.blit(self.background_img, (0, 0))

        x, y = 0, 0
        # Draw the Entire map
        while y < self.mid_h:
            x = 0
            while x < self.game.DISPLAY_W:
                # Calculate perspective (farther tiles are closer to 1)
                perspective = float(y / self.mid_w)

                midpoint = 0.5 + self.curvature * math.pow(1 - perspective, 3)
                road_w = 0.2 + perspective * 0.9
                clip_width = road_w * 0.16

                road_w *= 0.5

                LeftGrass = int((midpoint - road_w - clip_width) * self.game.DISPLAY_W)
                LeftClip = int((midpoint - road_w) * self.game.DISPLAY_W)
                RightClip = int((midpoint + road_w) * self.game.DISPLAY_W)
                RightGrass = int((midpoint + road_w + clip_width) * self.game.DISPLAY_W)

                nRow = self.mid_h + y

                # grass_color = (0,255,100)
                grass_color = (194, 178, 128)
                grass_val = math.sin(
                    20 * math.pow(1 - perspective, 3) + self.car.distance * 0.1
                )
                if grass_val > 0:
                    grass_color = (198, 163, 80)

                clip_color = (255, 0, 0)
                clip_val = math.sin(
                    80 * math.pow(1 - perspective, 2) + self.car.distance * 0.5
                )
                if clip_val > 0:
                    clip_color = (255, 255, 255)
                # Draw the appropriate tile
                if x >= 0 and x < LeftGrass:
                    pygame.draw.rect(self.game.display, grass_color, (x, nRow, 8, 8))
                if x >= LeftGrass and x < LeftClip:
                    pygame.draw.rect(self.game.display, clip_color, (x, nRow, 8, 8))
                if x >= LeftClip and x < RightClip:
                    pygame.draw.rect(self.game.display, (89, 89, 89), (x, nRow, 8, 8))
                if x >= RightClip and x < RightGrass:
                    pygame.draw.rect(self.game.display, clip_color, (x, nRow, 8, 8))
                if x >= RightGrass and x < self.game.DISPLAY_W:
                    pygame.draw.rect(self.game.display, grass_color, (x, nRow, 8, 8))
                x += 8
            y += 8

        # Draw the player's car
        self.car.draw()
        self.draw_stats()

    def draw_stats(self):
        speed_color = (255, 255, 255)
        if self.car.speed > 0.9:
            speed_color = (255, 0, 0)
        self.game.draw_text(
            "Speed (Km): " + str(round(self.car.speed * 100, 2)),
            20,
            speed_color,
            10,
            10,
        )
        self.game.draw_text(
            "Avg: " + str(round(self.game.get_average_speed() * 100, 2)),
            20,
            (255, 255, 255),
            10,
            30,
        )
        self.game.draw_text(
            "Time: " + str(round(self.game.lap_time, 2)), 20, (255, 255, 255), 10, 50
        )

        # Draw Action Arrows
        self.draw_action_arrows()

        if self.game.complete:
            self.game.draw_text(
                "FINISHED", 20, (255, 255, 255), self.mid_w - 10, self.mid_h
            )

        i = 0
        for lap in self.lap_times:
            self.game.draw_text(
                "Lap " + str(i + 1) + ": " + str(round(lap, 2)),
                20,
                (255, 255, 255),
                self.game.DISPLAY_W - 100,
                10 + (20 * i),
            )
            i += 1

    def draw_action_arrows(self):
        # Base position for the arrow group
        base_x, base_y = 10, 80
        size = 15
        spacing = 20

        actions = self.game.actions

        def draw_arrow(x, y, points, active):
            color = (0, 255, 0) if active else (50, 50, 50)
            # Offset points by x, y
            abs_points = [(p[0] + x, p[1] + y) for p in points]
            pygame.draw.polygon(self.game.display, color, abs_points)
            # Draw outline
            pygame.draw.polygon(self.game.display, (200, 200, 200), abs_points, 1)

        # UP (Accel)
        up_pts = [(size // 2, 0), (size, size), (0, size)]
        draw_arrow(base_x + spacing, base_y, up_pts, actions.get("accel", False))

        # DOWN (Brake)
        down_pts = [(0, 0), (size, 0), (size // 2, size)]
        draw_arrow(
            base_x + spacing, base_y + spacing, down_pts, actions.get("brake", False)
        )

        # LEFT
        left_pts = [(0, size // 2), (size, 0), (size, size)]
        draw_arrow(base_x, base_y + spacing, left_pts, actions.get("left", False))

        # RIGHT
        right_pts = [(size, size // 2), (0, 0), (0, size)]
        draw_arrow(
            base_x + spacing * 2, base_y + spacing, right_pts, actions.get("right", False)
        )

    def load_track(self):
        """Generate initial random track segments"""
        self.track = []
        self.track_length = 0
        # Generate initial track with enough segments
        self.generate_track_segments(20)

    def generate_track_segments(self, num_segments):
        """Generate random track segments and add them to the track"""
        for _ in range(num_segments):
            # Random curvature between -0.8 and 0.8
            # -0.8 = sharp left turn, 0 = straight, 0.8 = sharp right turn
            curvature = random.uniform(-0.8, 0.8)
            # Random segment length
            length = random.uniform(50, 200)
            self.track.append([curvature, length])
            self.track_length += length

    def update_track_generation(self, car_distance):
        """Check if we need to generate more track based on car distance"""
        # If car is getting close to the end of the generated track, add more
        if car_distance > self.track_length - 1000:
            self.generate_track_segments(10)

    def store_times(self):
        if self.lap_times:
            current_time = self.game.lap_time - sum(self.lap_times)
        else:
            current_time = self.game.lap_time
        self.lap_times.append(current_time)
