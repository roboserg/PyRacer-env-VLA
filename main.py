import pygame
from data.gameFiles.game import Game

def main():
    """Main entry point for PyRacer."""
    g = Game()

    try:
        while g.running:
            # If the player is not playing the game, show the appropriate menu
            g.current_menu.display_menu()
            
            # Clear event queue before starting game to avoid stale events
            pygame.event.clear()
            
            while g.playing:
                # Begin playing the game
                g.game_loop()
    except KeyboardInterrupt:
        print("\nGame exited by user.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
