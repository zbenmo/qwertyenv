from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Generator, List, Set, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Generator, List, Tuple


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "
COIN = '$'
DIAMOND = '*'

ROWS = '12345678'
COLS = 'abcdefgh'


Position = str # ex. 'e2'
PieceStr = str # ex. 'P'
Move = str # ex. 'e2e4' (UCI)

COINS_8 = COIN * 8
COINS_7 = COIN * 7

# Adapted from Chess
default_fen = f'r{COINS_7}/{COINS_8}/{COINS_8}/{COINS_8}/{COINS_8}/{COINS_8}/{COINS_8}/{COINS_7}R w'


@dataclass
class CollectCointsGameState:
    board: Dict[Position, PieceStr]
    turn: str
    castling_rights: set[bool]

    @staticmethod
    def from_fen(fen: str = default_fen) -> 'CollectCointsGameState':
        (
            positions,
            turn,
        ) = fen.split(' ')

        def _split_and_expand(line: str) -> Generator[str, None, None]:
            "helper generator function - from the compressed fen's board representation to full 8 characters row"
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield EMPTY
                else:
                    yield char

        board = {
            f'{col_n}{row_n}': piece_str
            for row_n, row in zip(reversed(ROWS), positions.split('/'))
            for col_n, piece_str in zip(COLS, _split_and_expand(row))
        }

        return CollectCointsGameState(
            board=board,
            turn=turn,
            castling_rights=set(),
        )

    def to_fen(self) -> str:

        def compress_and_join(line: List[str]) -> str:
            pieces = []
            count_empty = 0
            for piece in line:
                if piece != EMPTY:
                    if count_empty > 0:
                        pieces.append(str(count_empty))
                        count_empty = 0
                    pieces.append(piece)
                else:
                    count_empty += 1
            if count_empty > 0:
                pieces.append(str(count_empty))
            return ''.join(pieces)

        position = '/'.join(
            compress_and_join([self.board[f'{col}{row}'] for col in COLS])
            for row in reversed(ROWS)
        )
        return f'{position} {self.turn}'

    def _display_board(self, display_what: Callable[[Position], str]) -> None:
        for row in reversed(ROWS):
            for col in COLS:
                position = f'{col}{row}'
                print(display_what(position), end='')
            print()

    def _display_board_simple(self) -> None:
        self._display_board(lambda position: (
            self.board[position] if (self.board[position] != ' ') else '.')
        )

    def display(self) -> None:
        print()
        self._display_board_simple()
        print()
        print(f'turn={self.turn}')
        print()


@dataclass(repr=False)
class Square():
    "helper class to address a single square on the board"
    position: Position
    _row: int = field(init=False)
    _col: int = field(init=False)

    def __post_init__(self):
        col, row = self.position
        self._col = ord(col) - ord(COLS[0]) # 0 based
        self._row = ord(row) - ord(ROWS[0]) # 0 based

    @property
    def row(self) -> str:
        return self.position[1]

    @property
    def col(self) -> str:
        return self.position[0]

    def add(self, rows: int=0, cols: int=0) -> Position | None:
        row = self._row + rows
        col = self._col + cols
        if row < 0 or row > 7:
            return None
        if col < 0 or col > 7:
            return None
        return f'{chr(col + ord("a"))}{row + 1}'

    def __repr__(self):
        return self.position


IsChecked = Callable[[CollectCointsGameState], bool]


class Piece(ABC):
    def __init__(self, piece_str: PieceStr, position: Position):
        self._piece_str = piece_str
        self._square = Square(position)

    @abstractmethod
    def possible_moves(self, is_checked: IsChecked) -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
        pass

    @abstractmethod
    def positions_threats(self, game: CollectCointsGameState) -> Generator[Position, None, None]:
        pass

    def same_color(self, other_piece_str: PieceStr) -> bool:
        return self._piece_str.isupper() == other_piece_str.isupper()

    def attempt_move(self,
                     game: CollectCointsGameState,
                     move_square_str: Position,
                     is_checked: IsChecked,
                     promotion: str = None,
                     extra_from_to: Tuple[Position, Position] = None,
                     extra_remove: Position = None,
                     ) -> Tuple[Move, CollectCointsGameState] | None:
        """Note: The next_game_state, that is being returned, needs some additional working by the caller
        (board is correct yet other properties may need adjustment).
        """
        next_game_state = deepcopy(game)
        next_game_state.board[self._square.position], next_game_state.board[move_square_str] = (
            EMPTY, (promotion or self._piece_str)
        )
        if extra_from_to:
            # relevant for example for castling
            extra_from, extra_to = extra_from_to
            next_game_state.board[extra_from], next_game_state.board[extra_to] = (
                EMPTY, next_game_state.board[extra_from]
            )
        if extra_remove:
            # relevant for example for en-passant capture
            next_game_state.board[extra_remove] = EMPTY
        if is_checked(next_game_state):
            return None
        move = f'{self._square}{move_square_str}{promotion}' if promotion else f'{self._square}{move_square_str}'
        return move, next_game_state

    def update_castling_rights(self, next_game_state: CollectCointsGameState):
        """Just examines the board and updates the castling_rights.
        The logic ignores what just happened, as well as the previous casting_rights state.
        It (the logic) may be somewhat "wasteful", yet is straight-forward and is useful from various places in the code below. 
        """
        if next_game_state.board['a1'] != W_R:
            next_game_state.castling_rights.discard('Q')
        if next_game_state.board['h1'] != W_R:
            next_game_state.castling_rights.discard('K')
        if next_game_state.board['a8'] != B_R:
            next_game_state.castling_rights.discard('q')
        if next_game_state.board['h8'] != B_R:
            next_game_state.castling_rights.discard('k')


class Pawn(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_P or piece_str == B_P
        self._direction = +1 if self._piece_str == W_P else -1
        starting_row = ROWS[1] if self._piece_str == W_P else ROWS[-2]
        pre_promotion_row = ROWS[-2] if self._piece_str == W_P else ROWS[1]
        self.pre_promotion = pre_promotion_row == self._square.row
        self._theoretical_moves = [self._square.add(rows=1 * self._direction)]
        if self._square.row == starting_row:
            self._theoretical_moves.append(self._square.add(rows=2 * self._direction))
        self._theoretical_captures = [
            target_square_str
            for target_square_str in map(lambda x: self._square.add(rows=1 * self._direction, cols=x), [-1, +1])
            if target_square_str
        ]

    def possible_moves(self, game: CollectCointsGameState, is_checked: IsChecked) -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
        if self.pre_promotion:
            promotions = [W_Q, W_R, W_B, W_N] if self._piece_str == W_P else [B_Q, B_R, B_B, B_N]
            assert len(self._theoretical_moves) < 2 # ex. white can be on row 2 or on row 7, but not on both at the same time
        else:
            promotions = [None]

        def _move() -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
            two_steps = False
            for move_square_str in self._theoretical_moves:
                piece_there_str = game.board[move_square_str]
                if piece_there_str != EMPTY:
                    break
                for promotion in promotions:
                    ret = self.attempt_move(game, move_square_str, is_checked, promotion=promotion)
                    if not ret:
                        continue
                    move, next_game_state = ret
                    if self._piece_str == B_P:
                        assert game.turn == 'b'
                        next_game_state.move_number += 1
                        next_game_state.turn = 'w'
                    else:
                        assert game.turn == 'w'
                        next_game_state.turn = 'b'
                    next_game_state.en_passant = self._square.add(rows=1 * self._direction) if two_steps else '-'
                    next_game_state.half_moves = 0
                    yield move, next_game_state
                two_steps = True

        yield from _move()

        def _capture() -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
            for capture_square_str in self._theoretical_captures:
                piece_there_str = game.board[capture_square_str]
                extra_remove = None
                if piece_there_str == EMPTY:
                    if game.en_passant != capture_square_str:
                        continue
                    extra_remove = Square(capture_square_str).add(rows=-1 * self._direction)
                elif self.same_color(piece_there_str):
                    continue
                for promotion in promotions:
                    ret = self.attempt_move(
                        game, capture_square_str, is_checked, promotion=promotion, extra_remove=extra_remove)
                    if not ret:
                        continue
                    move, next_game_state = ret
                    if self._piece_str == B_P:
                        assert game.turn == 'b'
                        next_game_state.move_number += 1
                        next_game_state.turn = 'w'
                    else:
                        assert game.turn == 'w'
                        next_game_state.turn = 'b'
                    next_game_state.half_moves = 0
                    next_game_state.en_passant = '-'
                    self.update_castling_rights(next_game_state)
                    yield move, next_game_state

        yield from _capture()

    def positions_threats(self, game: CollectCointsGameState) -> Generator[Position, None, None]:
        yield from self._theoretical_captures


class Knight(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_N or piece_str == B_N
        self._theoretical_moves = [
            position
            for add_row, add_col in zip([1, 2], [2, 1])
            for sign_row, sign_col in product([-1, 1], [-1, 1])
            if (position := self._square.add(rows=sign_row * add_row, cols=sign_col * add_col)) is not None
        ]

    def possible_moves(self, game: CollectCointsGameState, is_checked: IsChecked) -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
        for move_square_str in self._theoretical_moves:
            piece_there_str = game.board[move_square_str]
            if piece_there_str != EMPTY and self.same_color(piece_there_str):
                continue
            ret = self.attempt_move(game, move_square_str, is_checked)
            if not ret:
                continue
            move, next_game_state = ret
            capture = piece_there_str != EMPTY
            if self._piece_str == B_N:
                assert game.turn == 'b'
                next_game_state.move_number += 1
                next_game_state.turn = 'w'
            else:
                assert game.turn == 'w'
                next_game_state.turn = 'b'
            next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
            if capture:
                self.update_castling_rights(next_game_state)
            next_game_state.en_passant = '-'
            yield move, next_game_state

    def positions_threats(self, game: CollectCointsGameState) -> Generator[Position, None, None]:
        yield from self._theoretical_moves


class RayBased(Piece, ABC):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        self._rays = [
            [target_square_str for target_square_str in ray if target_square_str]
            # if a target_square_str is "outside of the board" we shall have it here as None
            for ray in self.get_rays()
        ]

    @abstractmethod
    def get_rays(self) -> List[List[str]]:
        pass

    def possible_moves(self, game: CollectCointsGameState, is_checked: IsChecked) -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
        for ray in self._rays:
            for target_square_str in ray:
                piece_there_str = game.board[target_square_str]
                if piece_there_str != EMPTY and self.same_color(piece_there_str):
                    break # this ray reached a piece of its own color
                ret = self.attempt_move(game, target_square_str, is_checked)
                if not ret:
                    continue
                move, next_game_state = ret
                capture = piece_there_str != EMPTY
                if self.same_color(B_K):
                    assert game.turn == 'b'
                    next_game_state.move_number += 1
                    next_game_state.turn = 'w'
                else:
                    assert game.turn == 'w'
                    next_game_state.turn = 'b'
                next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
                if capture or self._piece_str in [W_R, B_R]:
                    self.update_castling_rights(next_game_state)
                next_game_state.en_passant = '-'
                yield move, next_game_state
                if capture:
                    break # this ray reached a piece (a capture)

    def positions_threats(self, game: CollectCointsGameState) -> Generator[Position, None, None]:
        for ray in self._rays:
            for target_square_str in ray:
                piece_there_str = game.board[target_square_str]
                yield target_square_str
                if piece_there_str != EMPTY:
                    break


class RookRays():
    "mixin"
    def _get_rook_rays(self) -> List[List[str]]:
        return [
            [self._square.add(cols=cols) for cols in range(1, 8)],
            [self._square.add(cols=cols) for cols in range(-1, -8, -1)],
            [self._square.add(rows=rows) for rows in range(1, 8)],
            [self._square.add(rows=rows) for rows in range(-1, -8, -1)],
        ]


class Rook(RayBased, RookRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_R or piece_str == B_R

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays()


class BishopRays():
    "mixin"
    def _get_bishop_rays(self) -> List[List[str]]:
        return [
            [self._square.add(r, c) for r, c in zip(range(1, 8), range(1, 8))],
            [self._square.add(r, c) for r, c in zip(range(-1, -8, -1), range(1, 8))],
            [self._square.add(r, c) for r, c in zip(range(1, 8), range(-1, -8, -1))],
            [self._square.add(r, c) for r, c in zip(range(-1, -8, -1), range(-1, -8, -1))],
        ]


class Bishop(RayBased, BishopRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_B or piece_str == B_B

    def get_rays(self) -> List[List[str]]:
        return self._get_bishop_rays()


class Queen(RayBased, RookRays, BishopRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_Q or piece_str == B_Q

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays() + self._get_bishop_rays()


class King(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_K or piece_str == B_K
        self._theoretical_moves = [
            position
            for add_rows, add_cols in product([-1, 0, 1], [-1, 0, 1]) if (
                (add_rows != 0 or add_cols != 0)
                and
                ((position := self._square.add(rows=add_rows, cols=add_cols)) is not None)
            )
        ]

    def possible_moves(self, game: CollectCointsGameState, is_checked: IsChecked) -> Generator[Tuple[Move, CollectCointsGameState],None,None]:

        def _simple() -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
            for move_square_str in self._theoretical_moves:
                piece_there_str = game.board[move_square_str]
                if piece_there_str != EMPTY and self.same_color(piece_there_str):
                    continue
                ret = self.attempt_move(game, move_square_str, is_checked)
                if not ret:
                    continue
                move, next_game_state = ret
                capture = piece_there_str != EMPTY
                if self._piece_str == B_K:
                    assert game.turn == 'b'
                    next_game_state.move_number += 1
                    next_game_state.castling_rights = next_game_state.castling_rights.difference(['k', 'q'])
                    next_game_state.turn = 'w'
                else:
                    assert game.turn == 'w'
                    next_game_state.castling_rights = next_game_state.castling_rights.difference(['K', 'Q'])
                    next_game_state.turn = 'b'

                next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
                if capture:
                    self.update_castling_rights(next_game_state)
                next_game_state.en_passant = '-'
                yield move, next_game_state

        yield from _simple()

        def _castling() -> Generator[Tuple[Move, CollectCointsGameState],None,None]:
            relevant_rights = set(['K', 'Q']) if self._piece_str == W_K else set(['k', 'q'])
            relevant_rights = relevant_rights.intersection(game.castling_rights)
            if len(relevant_rights) < 1:
                return
            checked_for_check = False
            for right in relevant_rights:
                match right:
                    case 'K':
                        assert self._square.position == 'e1'
                        assert game.board['h1'] == W_R
                        should_be_free = ['f1', 'g1']
                        additional_should_be_safe = ['f1'] # 'g1'
                        target_king = 'g1'
                        source_rook = 'h1'
                        target_rook = 'f1'
                    case 'Q':
                        assert self._square.position == 'e1'
                        assert game.board['a1'] == W_R
                        should_be_free = ['b1', 'c1', 'd1']
                        additional_should_be_safe = ['d1'] # 'c1'
                        target_king = 'c1'
                        source_rook = 'a1'
                        target_rook = 'd1'
                    case 'k':
                        assert self._square.position == 'e8'
                        assert game.board['h8'] == B_R
                        should_be_free = ['f8', 'g8']
                        additional_should_be_safe = ['f8'] # 'g8'
                        source_rook = 'h8'
                        target_king = 'g8'
                        target_rook = 'f8'
                    case 'q':
                        assert self._square.position == 'e8'
                        assert game.board['a8'] == B_R
                        should_be_free = ['b8', 'c8', 'd8']
                        additional_should_be_safe = ['d8'] # 'c8'
                        target_king = 'c8'
                        source_rook = 'a8'
                        target_rook = 'd8'
                    case _:
                        assert False, f'{right=}'
                if any(game.board[position] != EMPTY for position in should_be_free):
                    continue # this castling is not possible
                if not checked_for_check:
                    if is_checked(game):
                        return # no castling is possible
                    checked_for_check = True
                checked = None
                for position in additional_should_be_safe:
                    my_position = self._square.position
                    game.board[my_position], game.board[position] = EMPTY, game.board[my_position]
                    checked = is_checked(game)
                    # take back
                    game.board[position], game.board[my_position] = EMPTY, game.board[position]
                    if checked:
                        break
                if checked:
                    continue # this castling is not possible
                ret = self.attempt_move(game, target_king, is_checked, extra_from_to=(source_rook, target_rook))
                if not ret:
                    continue # this castling is not possible
                move, next_game_state = ret
                if self._piece_str == B_K:
                    assert game.turn == 'b'
                    next_game_state.move_number += 1
                    next_game_state.castling_rights = next_game_state.castling_rights.difference(['k', 'q'])
                    next_game_state.turn = 'w'
                else:
                    assert game.turn == 'w'
                    next_game_state.castling_rights = next_game_state.castling_rights.difference(['K', 'Q'])
                    next_game_state.turn = 'b'

                if next_game_state.castling_rights == '':
                    next_game_state.castling_rights = '-'

                next_game_state.half_moves = next_game_state.half_moves + 1
                next_game_state.en_passant = '-'
                yield move, next_game_state

        yield from _castling()

    def positions_threats(self, game: CollectCointsGameState) -> Generator[Position, None, None]:
        yield from self._theoretical_moves


class CollectCointsGame:

    @staticmethod
    @lru_cache
    def _piece_for(piece_str: PieceStr, position: Position) -> Piece:
        match piece_str:
            case 'p' | 'P':
                return Pawn(piece_str, position=position)
            case 'r' | 'R':
                return Rook(piece_str, position=position)
            case 'n' | 'N':
                return Knight(piece_str, position=position)
            case 'b' | 'B':
                return Bishop(piece_str, position=position)
            case 'q' | 'Q':
                return Queen(piece_str, position=position)
            case 'k' | 'K':
                return King(piece_str, position=position)
            case _:
                assert False, f'{piece_str=}'
        pass

    @staticmethod
    def is_checked(game: 'CollectCointsGame') -> bool:
        # which_king = W_K if game.turn == 'w' else B_K
        # kings_position = next(position for position, piece_str in game.board.items() if piece_str == which_king)
        # other_player = 'b' if game.turn == 'w' else 'w'
        # for piece_str, position in game.all_pieces(player=other_player):
        #     piece = CollectCointsGame._piece_for(piece_str, position)
        #     if any(kings_position == position for position in piece.positions_threats(game)):
        #         return True
        # return False
        return False

    def __init__(self):
        self._state = CollectCointsGameState.from_fen()
        self._scores = Counter()

    def step(self, move: str):
        from_position = move[:2]
        to_position = move[2:4]
        piece_str = self._state.board[from_position]
        assert piece_str != EMPTY # TODO: check a piece (not coin etc.)
        is_upper: bool = piece_str.isupper()
        assert (self._state.turn == 'b') or is_upper, f'{self._state.turn=}, {piece_str=}'
        piece = CollectCointsGame._piece_for(piece_str, from_position)
        for possible_move, next_game_state in piece.possible_moves(self._state, CollectCointsGame.is_checked):
            if possible_move == move:
                there = self._state.board[to_position]
                if there == COIN:
                    self._scores[self._state.turn] += 1
                elif there == DIAMOND:
                    self._scores[self._state.turn] += 5
                self._state = next_game_state
                return
        assert False, f'move {move} is not valid'

    def is_done(self) -> bool:
        return all(x not in [COIN, DIAMOND] for x in self._state.board.items()), 

    def display(self):
        self._state.display()
        print(self._scores)


def main():
    game_state = CollectCointsGameState.from_fen()

    game_state.display()

    print(game_state.to_fen())


def main_game():
    game = CollectCointsGame()

    game.display()

    print('-' * 80)

    game.step('h1g1')

    game.display()


if __name__ == "__main__":
    # main()
    main_game()