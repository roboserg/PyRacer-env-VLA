from data.gameFiles.game import Game
import pygame

g = Game()

while g.running:
    # If the player is not playing the game, show the appropriate menu
    g.current_menu.display_menu()
    # Clear event queue before starting game to avoid stale events
    pygame.event.clear()
    while g.playing:
        # Begin playing the game
        g.game_loop()
