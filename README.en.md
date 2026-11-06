# RSI Speedrun Ladder

[中文](README.md)

An AI coding agent (Claude Code) writes programs that play games, and a self-improvement loop keeps a new version only if it is faster. Each cell is one scored segment; a green cell means the agent's closed-loop policy finished all 5 unseen random variants (number = frames of the slowest run).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ladder.en.dark.svg">
  <img alt="RSI Speedrun Ladder" src="ladder.en.svg">
</picture>

| Rung | Game | | Segments | Cleared |
|---|---|---|---|---|
| 09 | Sekiro: Shadows Die Twice | real time on Windows | 8 | 4 |
| 08 | Minecraft | survival | 11 | 1 |
| 07 | Stardew Valley | 1.6 · real time on Windows | 9 | 4 |
| 06 | Quake | Easy skill | 8 | 0 |
| 05 | Doom | Ultra-Violence · pistol start | 9 | 0 |
| 04 | Celeste Classic | PICO-8 | 31 | 3 |
| 03 | Super Mario Bros.: The Lost Levels | Lost Levels | 16 | 1 |
| 02 | Super Mario Bros. | NES | 32 | 7 |
| 01 | MiniPlat | original platformer | 4 | 1 |

## Super Mario Bros. · closed-loop results (60 FPS)

| Level | Frames | Seconds | Variants |
|---|---|---|---|
| 1-1 | 1343 | 22.4 | 30/30 |
| 1-2 | 1347 | 22.4 | 5/5 |
| 1-3 | 1017 | 16.9 | 5/5 |
| 1-4 | 771 | 12.8 | 5/5 |
| 2-1 | 1389 | 23.1 | 5/5 |
| 2-2 | 2720 | 45.3 | 5/5 |
| 2-3 | 1339 | 22.3 | 5/5 |
| 2-4 | — | — | best 4/5 |

## Celeste Classic · open-loop results (30 FPS)

| Room | Frames | Seconds |
|---|---|---|
| 1 | 334 | 11.13 |
| 2 | 145 | 4.83 |
| 3 | 229 | 7.63 |

Rules: a closed-loop policy reads the screen pixels every frame and then chooses keys; no fixed key sequence. Games run in random mode and the referee runs the policy on 5 variants it has never seen; all must finish, and the slowest run is the score. Numbers are frames to finish; lower is better.

Game logos from Wikimedia Commons (mostly public domain; Celeste logo © Maddy Makes Games, CC BY-SA 4.0: https://commons.wikimedia.org/wiki/File:Celeste_video_game_logo.png). Trademarks belong to their owners and are used only to identify the games. Stardew Valley logo © ConcernedApe, taken from the game's cover art. The MiniPlat icon is original pixel art.
