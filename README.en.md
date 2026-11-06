# <img src="logos/emblem.svg" width="46" align="absmiddle" alt=""> RSI Game Speedrun Ladder

[中文](README.md)

An AI coding agent (Claude Code) writes programs that play games, and a self-improvement loop keeps a new version only if it is faster. Each cell is one scored segment; a green cell means the agent's closed-loop policy finished all 5 unseen random variants (number = frames of the slowest run).

## Dispatches

- **2026-11-06 11:40** **Stardew Valley**: into the Mines; the closed-loop bot breaks rocks, finds ladders and goes down from floor 1 to 3 (live on Windows, exploration save).
- **2026-11-06 11:19** **Sekiro**: from the wall corner into the grass, waiting out the talking guards, on to the crawl space (following a reference video, live on Windows).
- **2026-11-06 11:10** **Super Mario Bros. 2-4** (castle) cleared closed-loop: 1264 frames, 5/5 unseen variants. Worlds 1 and 2 are complete.
- **2026-11-06 10:00** **Super Mario Bros. 2-2** (underwater) cleared: 2720 frames, 5/5. 1-4 improved to 771 frames and 2-1 to 1389.
- **2026-11-06 08:28** **Super Mario Bros. 1-4, 2-1 and 2-3** cleared: 886 / 1760 / 1339 frames, all 5/5.
- **2026-11-06 07:41** **Stardew Valley**: a screen-only keyboard-and-mouse bot goes from a new save to the first 5 parsnips on Spring 5 (live on Windows).
- **2026-11-06 00:23** **Super Mario Bros. 1-3** cleared: 1017 frames, 5/5; 1-2 improved to 1347 frames.
- **2026-11-05 23:16** **Sekiro**: following reference frames from a human run, past the opening pit, the Ashina Reservoir rock face and the wall corner (live on Windows).
- **2026-11-05 22:54** **Super Mario Bros. 1-2** first closed-loop clear: 1545 frames, 5/5.
- **2026-11-05 09:30** **Super Mario Bros. 1-1**: first closed-loop clear in random mode, 1485 frames, 5/5 (1482 by noon).
- **2026-11-04 21:12** **Celeste Classic** rooms 1–3 cleared: 334 / 145 / 229 frames (30 FPS).
- **2026-11-04** First batch: **Minecraft** gets wood (402 frames); **Lost Levels** 1-1 cleared open-loop (1592 frames).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ladder.en.dark.svg">
  <img alt="RSI Game Speedrun Ladder" src="ladder.en.svg">
</picture>

| Rung | Game | | Segments | Cleared |
|---|---|---|---|---|
| 08 | Sekiro: Shadows Die Twice | real time on Windows | 8 | 5 |
| 07 | Minecraft | survival | 11 | 1 |
| 06 | Stardew Valley | 1.6 · real time on Windows | 9 | 5 |
| 05 | Quake | Easy skill | 8 | 0 |
| 04 | Doom (1993) | Ultra-Violence · pistol start | 9 | 0 |
| 03 | Celeste Classic | PICO-8 | 31 | 3 |
| 02 | Super Mario Bros.: The Lost Levels | NES ROM | 16 | 1 |
| 01 | Super Mario Bros. | NES | 32 | 8 |

## Super Mario Bros. · closed-loop results (60 FPS)

| Level | Frames | Seconds | Variants |
|---|---|---|---|
| 1-1 | 1482 | 24.7 | 5/5 |
| 1-2 | 1347 | 22.4 | 5/5 |
| 1-3 | 1017 | 16.9 | 5/5 |
| 1-4 | 771 | 12.8 | 5/5 |
| 2-1 | 1389 | 23.1 | 5/5 |
| 2-2 | 2720 | 45.3 | 5/5 |
| 2-3 | 1339 | 22.3 | 5/5 |
| 2-4 | 1264 | 21.1 | 5/5 |

## Celeste Classic · open-loop results (30 FPS)

| Room | Frames | Seconds |
|---|---|---|
| 1 | 334 | 11.13 |
| 2 | 145 | 4.83 |
| 3 | 229 | 7.63 |

Rules: a closed-loop policy reads the screen pixels every frame and then chooses keys; no fixed key sequence. Games run in random mode and the referee runs the policy on 5 variants it has never seen; all must finish, and the slowest run is the score. Numbers are frames to finish; lower is better.

Game logos from Wikimedia Commons (mostly public domain; Celeste logo © Maddy Makes Games, CC BY-SA 4.0: https://commons.wikimedia.org/wiki/File:Celeste_video_game_logo.png). Trademarks belong to their owners and are used only to identify the games. Stardew Valley logo © ConcernedApe, taken from the game's cover art. The ladder emblem is an original design.
