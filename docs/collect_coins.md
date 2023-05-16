# CollectCoins

CollectCoins is a Chess like game. Each of the two players, white and black, has one piece, a knight (one can also have rock vs. rock, or also additional combinations).
On the board squares there are coins.
Each player at its turn moves its piece to one of the valid places where the piece can move. If the target square had a coin, the coin is consumed by that player. In this game, pieces can not eat one another and cannot stand on the same square.
The aim of the game is to collect more coins than the other player, by planning better the path, by blocking the opponent, etc.

``` title="Shown below the initial board state and then the board after both players took their turn"
---------------------------------
|wR | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ |bR |
---------------------------------

0/0
---------------------------------
|   |wR | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ | $ | $ |
---------------------------------
| $ | $ | $ | $ | $ | $ |bR |   |
---------------------------------

1/1
```

In qwertyenv, at the moment, there are two implementations. One is a Gymnasium environment in which the other player takes a random (legal) move, and another is a PettingZoo environment. Both environments use a common module 'collect_coins_game.py'.

While experimenting with this environment, a few relevant wrappers were also developed, such as EnsureValidAction, that overrides your move if it is not legal, UpDownLeftRight that changes the action_space to {Up, Down, Left, Right}.

Also developed, and hopefully useful, are two utility functions to wrap (any) PettingZoo environment into the matching Gymnasium environment by declaring which is the "external agent", and where to fetch the action for the other agents.

The action space for both the Gymnasium and for the PettingZoo environments is the target square. This action spaces was choosen as to unify the action space with various pieces. My experience with this action space is, that it is slower to learn with it, than if the action space is specific for a piece (ex. {Up, Down, Left, Right}).

The observation space is the board (with the coins presence/absence), and the location of the two pieces.

``` py title="code snippet taken from the CollectCoins Gymnasium environment"
obs_space = dict(
  board = gym.spaces.Box(low=0, high=1, shape=(8 * 8,), dtype=bool),
  player = gym.spaces.MultiDiscrete([8, 8]),
  other_player = gym.spaces.MultiDiscrete([8, 8]),
)    
self.observation_space = gym.spaces.Dict(spaces=obs_space)
self.action_space = gym.spaces.MultiDiscrete([8, 8])
```

To initialize the environment pass the desired pieces (ex. 'rock' and 'rock'). Please note that a 'rock' here is a limited version that can only move one square at a time. You are expected also to indicate if you wish to go first or second (player=0 or player=1)
