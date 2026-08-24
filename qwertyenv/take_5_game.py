# This game is called also 6-Nimmt (in German).
# Base this implementation on https://github.com/johannbrehmer/rl-6-nimmt

from dataclasses import dataclass, field
import operator
import random
import numpy as np
from functools import reduce
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Player:
    cards: list = field(default_factory=list)
    negative_points : int = 0
    cards_collected: list = field(default_factory=list)

    def __str__(self):
        return '\n'.join([
            'cards in hand: ' + ', '.join(map(Take5Game._format_card, self.cards)),
            f'\tnegative points: {self.negative_points:>3d}'
        ])


class Take5Game:
    """ Take5Game (6-Nimmt) """

    def __init__(self, num_players: int, num_rows=4, num_cards=104, threshold=6):

        assert num_players > 0
        assert num_rows > 0
        assert num_cards >= 10 * num_players + num_rows
    
        self._players = [Player() for _ in range(num_players)]
        self._board = [[] for _ in range(num_rows)]
        self._threshold = threshold

        self._cards = list(range(1, num_cards + 1))
        random.shuffle(self._cards)

        for player in self._players:
            selected_cards, self._cards = self._cards[:10], self._cards[10:] # 10 cards per each player
            player.cards.extend(sorted(selected_cards))

        # Each row has always at least one card
        for row in self._board:
            row.append(self._cards.pop(0))

    def step(self, cards: list[int]):
        # logger.info('selected cards: ' + ', '.join(map(Take5Game._format_card, cards)))
        assert len(cards) == len(self._players)
        for o in np.argsort(cards):
            player, card = self._players[o], cards[o]
            player.cards.remove(card) # card must be there
            # now in which row the card goes?
            row_idx = None
            for i, row in enumerate(self._board):
                assert len(row) > 0
                if row[-1] > card:
                    continue
                if row_idx is None or self._board[row_idx][-1] < row[-1]:
                    row_idx = i
            if row_idx is None:
                # no row fits, user selects a row and takes the cards in that row (leaving the new card)
                min_idx = None
                min_val = None
                for i, row in enumerate(self._board):
                    val = reduce(operator.add, map(Take5Game._card_value, row))
                    if min_idx is None or val < min_val:
                        min_idx = i
                        min_val = val
                row_idx = min_idx
                self._take_cards(player, row_idx)
            else:
                if len(self._board[row_idx]) + 1 >= self._threshold:
                    self._take_cards(player, row_idx)
            self._board[row_idx].append(card)

    def is_done(self) -> bool:
        return len(self._players[0].cards) < 1

    def _take_cards(self, player, row_idx):
        player.negative_points += reduce(operator.add, map(Take5Game._card_value, self._board[row_idx]))
        player.cards_collected.extend(self._board[row_idx])
        self._board[row_idx].clear()

    def render(self):
        """ Report game progress somehow """

        logger.info("-" * 120)
        logger.info("Board:")
        for _, cards in enumerate(self._board):
            logger.info(f"  " + " ".join([self._format_card(card) for card in cards]) + "   _ " * (self._threshold - len(cards) - 1) + "   * ")
        logger.info("\nPlayers:")
        for p, player in enumerate(self._players):
            logger.info(
                f"{p})\t{player}\n"
            )
        if self.is_done():
            scores = [player.negative_points for player in self._players]
            winning_player = np.argmin(scores)
            losing_player = np.argmax(scores)
            logger.info(f"The game is over! {winning_player} wins, {losing_player} loses. Congratulations!")
        logger.info("-" * 120)

    @staticmethod
    def _card_value(card):
        """ Returns the points (Hornochsen) on a single card """

        assert 1 <= card <= 104, f'{card=}'

        if card == 55:
            return 7
        elif card % 11 == 0:  # 11, 22, ..., 99; but not 55
            return 5
        elif card % 10 == 0:  # 10, 20, 30, ..., 100
            return 3
        elif card % 10 == 5:  # 5, 15, ..., 95, but not 55
            return 2
        else:
            return 1

    @staticmethod
    def _format_card(card):
        signs = {1: " ", 2: ".", 3: ":", 5: "+", 7: "#"}
        value = Take5Game._card_value(card)
        return f"{card:>3d}{signs[value]}"


if __name__ == "__main__":
    take5 = Take5Game(3)
    take5.render()

    while not take5.is_done():
        logger.info('')
        selected_cards = [p.cards[0] for p in take5._players]
        take5.step(selected_cards)
        take5.render()
