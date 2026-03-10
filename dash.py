import pygame
import sys
import random

SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 400
FPS           = 60
GROUND_Y      = SCREEN_HEIGHT - 50

BLACK  = (10,  10,  20)
WHITE  = (240, 240, 255)
CYAN   = (0,   200, 255)
RED    = (255, 60,  60)
ORANGE = (255, 160, 40)
PURPLE = (180, 60,  255)
GRAY   = (90,  100, 115)

PLAYER_SIZE = 28
PLATFORM_H  = 12
SPIKE_SIZE  = 34

# Top surface Y of each elevated level
LEVEL_1_Y = GROUND_Y - 95
LEVEL_2_Y = GROUND_Y - 165

# Pre-computed spike Y for each lane (spikes sit flush on their surface)
SPIKE_Y = [
    GROUND_Y  - SPIKE_SIZE,   # lane 0 = ground
    LEVEL_1_Y - SPIKE_SIZE,   # lane 1 = L1 platform
    LEVEL_2_Y - SPIKE_SIZE,   # lane 2 = L2 platform
]

BASE_SPEED = 6
MAX_SPEED  = 13

# No obstacles for first SAFE_ZONE px of a chunk (player onboarding / reaction)
SAFE_ZONE = 100


# ── Player ────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self):
        self.size  = PLAYER_SIZE
        self.x     = 100
        self.y     = GROUND_Y - self.size
        self.vel_y = 0
        self.grav  = 0.65
        self.jv    = -13
        self.max_j = 2
        self.jumps = self.max_j

    def jump(self):
        if self.jumps > 0:
            self.vel_y = self.jv
            self.jumps -= 1

    def update(self, plats, scroll):
        self.vel_y += self.grav
        self.y     += self.vel_y

        # Ground collision
        if self.y + self.size >= GROUND_Y:
            self.y     = GROUND_Y - self.size
            self.vel_y = 0
            self.jumps = self.max_j
            return

        # One-way platform collision: land from above only.
        # Player can jump upward through platforms to reach higher levels.
        for p in plats:
            sx = p.x - scroll
            if sx + p.width < self.x or sx > self.x + self.size:
                continue
            prev_bot = self.y + self.size - self.vel_y
            cur_bot  = self.y + self.size
            if self.vel_y > 0 and prev_bot <= p.y + 1 and cur_bot >= p.y:
                self.y     = p.y - self.size
                self.vel_y = 0
                self.jumps = self.max_j

    def hitbox(self):
        m = 5
        return pygame.Rect(self.x + m, self.y + m,
                           self.size - 2*m, self.size - 2*m)

    def draw(self, surf):
        c = CYAN if self.jumps == 2 else (PURPLE if self.jumps == 1 else GRAY)
        pygame.draw.rect(surf, c, (self.x, self.y, self.size, self.size))
        pygame.draw.rect(surf, WHITE, (self.x, self.y, self.size, self.size), 1)
        pygame.draw.rect(surf, WHITE, (self.x + 19, self.y + 7, 5, 5))


# ── Terrain ───────────────────────────────────────────────────────────────────

class Platform:
    def __init__(self, x, y, w):
        self.x, self.y, self.width = x, y, w

    def draw(self, surf, scroll):
        sx = self.x - scroll
        if sx + self.width < 0 or sx > SCREEN_WIDTH:
            return
        pygame.draw.rect(surf, GRAY, (sx, self.y, self.width, PLATFORM_H))
        pygame.draw.line(surf, WHITE,
                         (max(sx, 0), self.y),
                         (min(sx + self.width, SCREEN_WIDTH), self.y), 2)


class Spike:
    def __init__(self, x, y, color=RED):
        self.x, self.y = x, y
        self.color     = color
        self.width     = SPIKE_SIZE

    def draw(self, surf, scroll):
        sx = self.x - scroll
        if sx + SPIKE_SIZE < 0 or sx > SCREEN_WIDTH:
            return
        s = SPIKE_SIZE
        pygame.draw.polygon(surf, self.color,
                             [(sx, self.y + s),
                              (sx + s // 2, self.y),
                              (sx + s, self.y + s)])

    def hitbox(self, scroll):
        sx = self.x - scroll
        s  = SPIKE_SIZE
        return pygame.Rect(sx + s//3, self.y + s//2, s//3, s//2)


# ── Procedural generation ─────────────────────────────────────────────────────
#
# A chunk is a sequence of obstacle sections separated by platform gaps.
#
# Each SECTION:
#   • Has a platform at L1 and L2 (both always present, so no floating spikes).
#   • Fully blocks one lane (dense RED spikes — must use a different lane).
#   • Lightly blocks a second lane (sparse ORANGE spikes spaced far apart —
#     player can jump over individual spikes while staying on that lane).
#   • Leaves one lane completely clear.
#
# Each GAP between sections (55-80 px of no platforms, no spikes):
#   • L2 player falls ~15 px → still above L1, lands on next section's L1 platform.
#   • L1 player falls ~15 px → drops below L1, falls to ground before next L1.
#   • Ground player stays on ground.
#   This makes gaps natural level-transition zones without any special logic.
#
# Speed increases with score, capped at MAX_SPEED.

def build_chunk(start_x):
    plats, spikes = [], []

    x         = start_x + SAFE_ZONE
    chunk_end = start_x + 520
    blocked   = random.randint(0, 2)

    while x < chunk_end - 20:
        sec_len = random.randint(90, 200)
        sec_end = min(x + sec_len, chunk_end - 20)
        sec_w   = sec_end - x

        # Platforms for this section (L1 and L2 only — ground is always present)
        plats.append(Platform(x, LEVEL_1_Y, sec_w))
        plats.append(Platform(x, LEVEL_2_Y, sec_w))

        # Pick partial (orange) and clear lane
        others  = [l for l in range(3) if l != blocked]
        random.shuffle(others)
        partial = others[0]
        # others[1] is the completely clear lane

        # Dense red spikes — wall across the full section length
        dense_step = SPIKE_SIZE + random.randint(3, 8)
        xi = x + 10   # tiny entry margin
        while xi + SPIKE_SIZE <= sec_end:
            spikes.append(Spike(xi, SPIKE_Y[blocked], color=RED))
            xi += dense_step

        # Sparse orange spikes — spaced ~90-120 px so each one is individually
        # jumpable; player can stay on this lane by hopping over them.
        sparse_step = SPIKE_SIZE + random.randint(60, 90)
        xi = x + random.randint(25, 50)   # randomised start offset
        while xi + SPIKE_SIZE <= sec_end:
            spikes.append(Spike(xi, SPIKE_Y[partial], color=ORANGE))
            xi += sparse_step

        # Gap between sections: no platforms, no spikes.
        # Physics: in ~10 frames of free-fall an L1 player descends ~30 px,
        # dropping below the next section's L1 platform and landing on ground.
        # An L2 player descends ~30 px, still above L1, landing on next L1.
        gap = random.randint(55, 80)
        x   = sec_end + gap

        # Change blocked lane with 60 % probability
        if random.random() < 0.60:
            blocked = random.choice([l for l in range(3) if l != blocked])

    return plats, spikes


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Infinite Dash — 3 Lanes")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Arial", 20, bold=True)
    sfont = pygame.font.SysFont("Arial", 13)

    state = {}

    def spawn():
        p, s = build_chunk(state["next_x"])
        state["plats"].extend(p)
        state["spikes"].extend(s)
        state["next_x"] += 520 + random.randint(60, 130)

    def restart():
        state["player"]  = Player()
        state["plats"]   = []
        state["spikes"]  = []
        state["scroll"]  = 0
        state["next_x"]  = 650
        state["dead"]    = False
        state["score_t"] = pygame.time.get_ticks()
        for _ in range(3):
            spawn()

    restart()

    while True:
        clock.tick(FPS)
        screen.fill(BLACK)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_SPACE,
                                                        pygame.K_UP,
                                                        pygame.K_w):
                if state["dead"]:
                    restart()
                else:
                    state["player"].jump()

        p    = state["player"]
        dead = state["dead"]
        sc   = state["scroll"]

        if not dead:
            score = (pygame.time.get_ticks() - state["score_t"]) // 100
            speed = min(BASE_SPEED + score // 25, MAX_SPEED)
            state["scroll"] += speed
            sc = state["scroll"]

            if state["next_x"] - sc < SCREEN_WIDTH + 500:
                spawn()

            state["plats"]  = [o for o in state["plats"]
                                if o.x + o.width - sc > -60]
            state["spikes"] = [o for o in state["spikes"]
                                if o.x + o.width - sc > -60]

            p.update(state["plats"], sc)

            for sp in state["spikes"]:
                if p.hitbox().colliderect(sp.hitbox(sc)):
                    state["dead"] = True
                    dead = True
                    break
        else:
            score = 0
            speed = 0

        # Level-zone shading (subtle background bands)
        pygame.draw.rect(screen, (18, 22, 40),
                         (0, LEVEL_2_Y - SPIKE_SIZE,
                          SCREEN_WIDTH, SPIKE_SIZE + PLATFORM_H))
        pygame.draw.rect(screen, (25, 15, 40),
                         (0, LEVEL_1_Y - SPIKE_SIZE,
                          SCREEN_WIDTH, SPIKE_SIZE + PLATFORM_H))

        pygame.draw.line(screen, WHITE, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 3)

        for obj in state["plats"]:
            obj.draw(screen, sc)
        for obj in state["spikes"]:
            obj.draw(screen, sc)

        if not dead:
            p.draw(screen)

        if not dead:
            screen.blit(font.render(f"SCORE: {score}  SPD: {speed}", True, WHITE),
                        (20, 14))
            screen.blit(sfont.render("L2", True, (80, 80, 130)), (6, LEVEL_2_Y - 14))
            screen.blit(sfont.render("L1", True, (80, 80, 130)), (6, LEVEL_1_Y - 14))
        else:
            msg = font.render("YOU DIED  —  SPACE to restart", True, RED)
            screen.blit(msg, (SCREEN_WIDTH//2 - msg.get_width()//2,
                               SCREEN_HEIGHT//2 - 14))

        pygame.display.flip()


if __name__ == "__main__":
    main()
