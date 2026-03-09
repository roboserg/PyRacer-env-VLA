import pygame
import sys
import random

# --- Configuration ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60
GROUND_Y = SCREEN_HEIGHT - 70

# --- Colors ---
BLACK = (15, 15, 15)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 50, 50)
PURPLE = (180, 50, 255)
GRAY = (50, 50, 50)

class Player:
    def __init__(self):
        self.size = 30
        self.x = 100
        self.y = GROUND_Y - self.size
        self.vel_y = 0
        self.gravity = 0.7
        self.jump_strength = -11
        self.max_jumps = 2
        self.jumps_left = self.max_jumps

    def jump(self):
        if self.jumps_left > 0:
            self.vel_y = self.jump_strength
            self.jumps_left -= 1

    def update(self):
        self.vel_y += self.gravity
        self.y += self.vel_y

        # Floor Collision
        if self.y > GROUND_Y - self.size:
            self.y = GROUND_Y - self.size
            self.vel_y = 0
            self.jumps_left = self.max_jumps

    def draw(self, screen):
        color = CYAN if self.jumps_left == 2 else (PURPLE if self.jumps_left == 1 else GRAY)
        pygame.draw.rect(screen, color, (self.x, self.y, self.size, self.size))
        # Add a little eye to show direction
        pygame.draw.rect(screen, WHITE, (self.x + 20, self.y + 5, 5, 5))

class GameObject:
    def __init__(self, x, y, width, height, type):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type # "block" or "spike"

    def draw(self, screen, scroll):
        rect = (self.x - scroll, self.y, self.width, self.height)
        if self.type == "block":
            pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)
        elif self.type == "spike":
            points = [
                (rect[0], rect[1] + rect[3]),
                (rect[0] + rect[2]/2, rect[1]),
                (rect[0] + rect[2], rect[1] + rect[3])
            ]
            pygame.draw.polygon(screen, RED, points)

def main():
    # Selective initialization to bypass audio hangs in WSL
    pygame.display.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Infinite Dash")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24, bold=True)

    player = Player()
    obstacles = []
    
    scroll = 0
    speed = 6
    last_obstacle_x = 600
    start_ticks = pygame.time.get_ticks() # For score tracking

    run = True
    while run:
        screen.fill(BLACK)
        
        # --- Score Calculation ---
        # 1 point per 100ms
        current_ticks = pygame.time.get_ticks()
        score = (current_ticks - start_ticks) // 100

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_SPACE, pygame.K_UP, pygame.K_w]:
                    player.jump()

        # --- Procedural Generation ---
        # If the furthest obstacle is getting close to the screen edge, make a new one
        if last_obstacle_x - scroll < SCREEN_WIDTH + 200:
            spawn_type = random.choice(["spike", "double_spike", "platform", "high_platform", "gap"])
            
            if spawn_type == "spike":
                obstacles.append(GameObject(last_obstacle_x, GROUND_Y - 40, 40, 40, "spike"))
            elif spawn_type == "double_spike":
                obstacles.append(GameObject(last_obstacle_x, GROUND_Y - 40, 80, 40, "spike"))
            elif spawn_type == "platform":
                obstacles.append(GameObject(last_obstacle_x, GROUND_Y - 100, 120, 20, "block"))
                if random.random() > 0.6: # Chance for a spike on top
                    obstacles.append(GameObject(last_obstacle_x + 40, GROUND_Y - 140, 40, 40, "spike"))
            elif spawn_type == "high_platform":
                obstacles.append(GameObject(last_obstacle_x, GROUND_Y - 180, 120, 20, "block"))

            last_obstacle_x += random.randint(250, 450) # Distance to next obstacle cluster

        # --- Updates ---
        player.update()
        scroll += speed

        # --- Cleanup ---
        # Remove obstacles that are far off-screen to save memory
        obstacles = [obj for obj in obstacles if obj.x - scroll > -200]

        # --- Drawing & Collisions ---
        pygame.draw.line(screen, WHITE, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 3)
        
        player_rect = pygame.Rect(player.x, player.y, player.size, player.size)
        
        for obj in obstacles:
            obj.draw(screen, scroll)
            obj_rect = pygame.Rect(obj.x - scroll, obj.y, obj.width, obj.height)

            if player_rect.colliderect(obj_rect):
                if obj.type == "spike":
                    # Death: Reset everything
                    scroll = 0
                    last_obstacle_x = 600
                    obstacles = []
                    player.y = GROUND_Y - player.size
                    player.vel_y = 0
                    start_ticks = pygame.time.get_ticks()
                
                elif obj.type == "block":
                    # Check if landing on top
                    if player.vel_y > 0 and player.y + player.size < obj.y + 20:
                        player.y = obj.y - player.size
                        player.vel_y = 0
                        player.jumps_left = player.max_jumps
                    else:
                        # Hit side of block: Death
                        scroll = 0
                        last_obstacle_x = 600
                        obstacles = []
                        player.y = GROUND_Y - player.size
                        start_ticks = pygame.time.get_ticks()

        player.draw(screen)

        # --- Draw Score UI ---
        score_surface = font.render(f"SCORE: {score}", True, WHITE)
        screen.blit(score_surface, (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
