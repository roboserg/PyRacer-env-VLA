import pygame
from data.gameFiles.game import Game

def main():
    """Main entry point for PyRacer."""
    g = Game()
    try:
        g.game_loop()
    except KeyboardInterrupt:
        print("\nGame exited by user.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
