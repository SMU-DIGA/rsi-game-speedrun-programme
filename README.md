# RSI Game Speedrun Ladder

RSI 游戏速通天梯

[English](README.en.md)

让 AI 编程智能体（Claude Code）为游戏写通关程序，并在“只有更快才保留”的自我改进循环里不断刷新成绩。每一格是一个评测分段；绿色格子是智能体写的闭环策略在 5 个没见过的随机变体上全部通关的成绩（帧数，取最慢一局）。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ladder.dark.svg">
  <img alt="RSI Game Speedrun Ladder · 游戏速通天梯" src="ladder.svg">
</picture>

| 档 | 游戏 | 说明 | 分段 | 已通关 |
|---|---|---|---|---|
| 09 | 只狼：影逝二度 Sekiro: Shadows Die Twice | Windows 实时 | 8 | 4 |
| 08 | 我的世界 Minecraft | 生存模式 | 11 | 1 |
| 07 | 星露谷物语 Stardew Valley | 1.6 · Windows 实时 | 9 | 4 |
| 06 | 雷神之锤 Quake | Easy 难度 | 8 | 0 |
| 05 | 毁灭战士 Doom (1993) | UV 难度 · 手枪开局 | 9 | 0 |
| 04 | 蔚蓝经典版 Celeste Classic | PICO-8 | 31 | 3 |
| 03 | 超级马里奥兄弟：失落的关卡 Super Mario Bros.: The Lost Levels | NES ROM 版 | 16 | 1 |
| 02 | 超级马里奥兄弟 Super Mario Bros. | NES | 32 | 7 |
| 01 | MiniPlat | 原创平台跳跃 | 4 | 1 |

## 超级马里奥兄弟 · 闭环成绩（60 FPS）

| 关卡 | 帧数 | 秒 | 变体 |
|---|---|---|---|
| 1-1 | 1343 | 22.4 | 30/30 |
| 1-2 | 1347 | 22.4 | 5/5 |
| 1-3 | 1017 | 16.9 | 5/5 |
| 1-4 | 771 | 12.8 | 5/5 |
| 2-1 | 1389 | 23.1 | 5/5 |
| 2-2 | 2720 | 45.3 | 5/5 |
| 2-3 | 1339 | 22.3 | 5/5 |
| 2-4 | — | — | 最好 4/5 |

## 蔚蓝经典版 · 开环成绩（30 FPS）

| 房间 | 帧数 | 秒 |
|---|---|---|
| 1 | 334 | 11.13 |
| 2 | 145 | 4.83 |
| 3 | 229 | 7.63 |

规则：闭环策略每帧读取屏幕像素再决定按键，不使用固定按键序列；游戏开启随机模式，裁判在 5 个没见过的变体上运行，全部通关才算，成绩取最慢一局。格内数字是通关帧数，越小越好。

游戏标志来自 Wikimedia Commons（多为公有领域；Celeste 标志 © Maddy Makes Games，CC BY-SA 4.0：https://commons.wikimedia.org/wiki/File:Celeste_video_game_logo.png），商标归各自权利人所有，仅用于标识游戏。星露谷物语标志 © ConcernedApe，取自游戏封面。MiniPlat 图标为原创像素画。
