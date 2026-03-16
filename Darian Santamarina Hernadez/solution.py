import random
import math
import time
from collections import deque
from player import Player
from board import HexBoard

# --- EXCEPCIONES Y FUNCIONES GLOBALES ---

class TimeoutException(Exception):
    pass

def get_neighbors(r, c, size):
    """Precomputar vecinos fuera de las clases para mayor velocidad"""
    if r % 2 == 0:
        directions = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
    else:
        directions = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
    
    neighbors = []
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors

# --- CLASES AUXILIARES (MCTS) ---

class Node:
    __slots__ = ['move', 'parent', 'player_at_node', 'children', 'wins', 'visits', 'untried_moves']
    
    def __init__(self, move=None, parent=None, player_at_node=None, untried_moves=None):
        self.move = move
        self.parent = parent
        self.player_at_node = player_at_node  
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = untried_moves

    def select_child(self, explore_param):
        log_parent = math.log(self.visits)
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            score = (child.wins / child.visits) + explore_param * math.sqrt(log_parent / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def add_child(self, move, player, untried_moves):
        child = Node(move=move, parent=self, player_at_node=player, untried_moves=untried_moves)
        self.children.append(child)
        return child

# --- IA MAESTRA HÍBRIDA ---

class MasterPlayer(Player):
    def __init__(self, player_id: int, time_limit: float = 4.0, max_depth: int = 4, switch_threshold: int = 7):
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1
        self.time_limit = time_limit
        
        # Parámetros Minimax
        self.base_depth = max_depth
        self.memo = {}
        
        # Parámetros MCTS
        self.neighbors_cache = {}
        
        # Umbral de decisión (<= 7 usa Minimax, > 7 usa MCTS)
        self.switch_threshold = switch_threshold

    def play(self, board: HexBoard) -> tuple:
        """El cerebro maestro que decide qué algoritmo ejecutar"""
        self.start_time = time.time()
        
        if board.size <= self.switch_threshold:
            # print(f"MasterPlayer: Usando MINIMAX (Tablero {board.size}x{board.size})")
            return self._play_minimax(board)
        else:
            # print(f"MasterPlayer: Usando MONTECARLO (Tablero {board.size}x{board.size})")
            return self._play_mcts(board)

    # ==========================================
    #             LÓGICA MINIMAX
    # ==========================================
    
    def _play_minimax(self, board: HexBoard) -> tuple:
        size = board.size
        self.memo = {}
        empty_cells = self._get_empty_cells(board)

        if len(empty_cells) >= (size * size) - 1:
            center = size // 2
            if board.board[center][center] == 0: return (center, center)

        my_path_nodes = self._get_shortest_path_nodes(board, self.player_id)
        opp_path_nodes = self._get_shortest_path_nodes(board, self.opponent_id)
        intersection = my_path_nodes.intersection(opp_path_nodes)

        def move_priority(cell):
            score = 0
            if cell in intersection: score -= 3000
            elif cell in my_path_nodes: score -= 1500
            elif cell in opp_path_nodes: score -= 1000
            score += abs(cell[0] - size//2) + abs(cell[1] - size//2)
            return score

        empty_cells.sort(key=move_priority)
        candidate_moves = empty_cells if size <= 7 else empty_cells[:12]
        best_move_global = candidate_moves[0]
        
        try:
            target_limit = 20 if size <= 5 else self.base_depth + 4
            
            for depth in range(1, target_limit + 1):
                current_best_score = -math.inf
                alpha = -math.inf
                beta = math.inf
                temp_best_move = None
                
                for r, c in candidate_moves:
                    if time.time() - self.start_time > self.time_limit:
                        raise TimeoutException
                    
                    board.board[r][c] = self.player_id
                    score = self._minimax_algo(board, depth - 1, alpha, beta, False)
                    board.board[r][c] = 0

                    if score > current_best_score:
                        current_best_score = score
                        temp_best_move = (r, c)
                    
                    alpha = max(alpha, current_best_score)
                    if beta <= alpha: break
                
                if temp_best_move:
                    best_move_global = temp_best_move
                
                if current_best_score >= 10000: break

        except TimeoutException:
            pass

        return best_move_global

    def _minimax_algo(self, board: HexBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutException

        if board.check_connection(self.player_id): return 10000 + depth
        if board.check_connection(self.opponent_id): return -10000 - depth

        if depth == 0:
            state_key = tuple(tuple(row) for row in board.board)
            if state_key in self.memo: return self.memo[state_key]
            
            val = self._evaluate_board(board)
            self.memo[state_key] = val
            return val

        empty_cells = self._get_empty_cells(board)
        if not empty_cells: return 0

        branch_limit = 10 if board.size > 7 else 25

        if is_maximizing:
            max_eval = -math.inf
            for r, c in empty_cells[:branch_limit]:
                board.board[r][c] = self.player_id
                eval_score = self._minimax_algo(board, depth - 1, alpha, beta, False)
                board.board[r][c] = 0
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = math.inf
            for r, c in empty_cells[:branch_limit]:
                board.board[r][c] = self.opponent_id
                eval_score = self._minimax_algo(board, depth - 1, alpha, beta, True)
                board.board[r][c] = 0
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def _evaluate_board(self, board: HexBoard) -> float:
        my_dist = self._shortest_path_value(board, self.player_id)
        opp_dist = self._shortest_path_value(board, self.opponent_id)
        
        if opp_dist == 0: return -10000
        if my_dist == 0: return 10000
        
        return (opp_dist * 25) - (my_dist * 60)

    def _shortest_path_value(self, board: HexBoard, player: int) -> int:
        size = board.size
        queue = deque()
        visited = {} 
        opp = 2 if player == 1 else 1

        if player == 1: 
            for r in range(size):
                if board.board[r][0] == opp: continue
                cost = 0 if board.board[r][0] == player else 1
                queue.append((r, 0, cost))
                visited[(r, 0)] = cost
        else: 
            for c in range(size):
                if board.board[0][c] == opp: continue
                cost = 0 if board.board[0][c] == player else 1
                queue.append((0, c, cost))
                visited[(0, c)] = cost

        min_cost = 100
        while queue:
            r, c, cost = queue.popleft()
            if cost >= min_cost: continue
            if (player == 1 and c == size - 1) or (player == 2 and r == size - 1):
                min_cost = min(min_cost, cost)
                continue

            for nr, nc in get_neighbors(r, c, size):
                if board.board[nr][nc] == opp: continue
                new_cost = cost + (0 if board.board[nr][nc] == player else 1)
                if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                    visited[(nr, nc)] = new_cost
                    if board.board[nr][nc] == player: queue.appendleft((nr, nc, new_cost))
                    else: queue.append((nr, nc, new_cost))
        return min_cost

    def _get_shortest_path_nodes(self, board: HexBoard, player: int) -> set:
        size = board.size
        queue = deque()
        visited = {} 
        opp = 2 if player == 1 else 1

        if player == 1:
            for r in range(size):
                if board.board[r][0] == opp: continue
                cost = 0 if board.board[r][0] == player else 1
                queue.append((r, 0, cost, {(r, 0)}))
                visited[(r, 0)] = cost
        else:
            for c in range(size):
                if board.board[0][c] == opp: continue
                cost = 0 if board.board[0][c] == player else 1
                queue.append((0, c, cost, {(0, c)}))
                visited[(0, c)] = cost

        min_cost = 100
        best_path = set()
        while queue:
            r, c, cost, path = queue.popleft()
            if cost > min_cost: continue
            if (player == 1 and c == size - 1) or (player == 2 and r == size - 1):
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
                continue

            for nr, nc in get_neighbors(r, c, size):
                if board.board[nr][nc] == opp: continue
                new_cost = cost + (0 if board.board[nr][nc] == player else 1)
                if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                    visited[(nr, nc)] = new_cost
                    new_path = path | {(nr, nc)}
                    if board.board[nr][nc] == player: queue.appendleft((nr, nc, new_cost, new_path))
                    else: queue.append((nr, nc, new_cost, new_path))
        return best_path

    def _get_empty_cells(self, board: HexBoard) -> list:
        return [(r, c) for r in range(board.size) for c in range(board.size) if board.board[r][c] == 0]

    # ==========================================
    #             LÓGICA MONTECARLO
    # ==========================================

    def _play_mcts(self, board: HexBoard) -> tuple:
        size = board.size
        explore_param = 1.41 if size <= 11 else 0.9
        
        if not self.neighbors_cache:
            for r in range(size):
                for c in range(size):
                    self.neighbors_cache[(r, c)] = get_neighbors(r, c, size)

        empty_cells = self._get_empty_cells(board)
        if not empty_cells:
            return None
            
        root = Node(player_at_node=self.opponent_id, untried_moves=empty_cells[:])
        end_time = self.start_time + self.time_limit
        iterations = 0

        while time.time() < end_time:
            node = root
            state = [row[:] for row in board.board] 
            current_empty = empty_cells[:]

            # Selección
            while not node.untried_moves and node.children:
                node = node.select_child(explore_param)
                r, c = node.move
                state[r][c] = node.player_at_node
                current_empty.remove(node.move)

            # Expansión
            current_player = 1 if node.player_at_node == 2 else 2
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                node.untried_moves[idx], node.untried_moves[-1] = node.untried_moves[-1], node.untried_moves[idx]
                move = node.untried_moves.pop()
                
                r, c = move
                state[r][c] = current_player
                current_empty.remove(move)
                
                child_untried = current_empty[:]
                node = node.add_child(move, current_player, child_untried)

            # Simulación
            winner = self._fast_playout(state, size, 1 if node.player_at_node == 2 else 2, current_empty)

            # Retropropagación
            while node is not None:
                node.visits += 1
                if node.player_at_node == winner:
                    node.wins += 1
                node = node.parent
            
            iterations += 1

        print(f"MasterPlayer (MCTS): {iterations} simulaciones en {time.time() - self.start_time:.2f}s")
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _fast_playout(self, state, size, current_player, empty_cells):
        random.shuffle(empty_cells)
        for r, c in empty_cells:
            state[r][c] = current_player
            current_player = 1 if current_player == 2 else 2
        return self._check_winner_fast(state, size)

    def _check_winner_fast(self, state, size):
        queue = deque()
        for r in range(size):
            if state[r][0] == 1:
                queue.append((r, 0))
                state[r][0] = 3 
        
        while queue:
            r, c = queue.popleft()
            if c == size - 1: 
                return 1
            
            for nr, nc in self.neighbors_cache[(r, c)]:
                if state[nr][nc] == 1:
                    state[nr][nc] = 3 
                    queue.append((nr, nc))
                    
        return 2