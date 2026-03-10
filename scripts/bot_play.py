#!/usr/bin/env python3
from vla.env import GameEnvironment
from vla.agents.bot_agent import BotAgent

def main():
    agent = BotAgent()
    env = GameEnvironment(agent, recorder=None)
    stats = env.run()
    print(f"Final speed: {stats['final_speed']:.2f}, Steps: {stats['steps']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
