from player import Player
from board import HexBoard
from collections import deque
import math

class SmartPlayer(Player):
    def __init__(self, player_id: int, max_depth: int = 2):
        """
        Inicializa el jugador autónomo.
        max_depth: Profundidad base. Se ajustará dinámicamente según el tamaño del tablero.
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1
        self.base_depth = max_depth

    def play(self, board: HexBoard) -> tuple:
        size = board.size
        empty_cells = self.get_empty_cells(board)

        # 1. Ajuste dinámico de profundidad
        current_depth = self.base_depth
        if size <= 5: current_depth = max(4, self.base_depth)
        elif size <= 7: current_depth = max(3, self.base_depth)

        # 2. Apertura central
        if len(empty_cells) >= (size * size) - 1:
            center = size // 2
            if board.board[center][center] == 0: return (center, center)

        # 3. ORDENACIÓN DE MOVIMIENTOS: Priorizamos fuertemente nuestro ataque
        my_path = self.get_shortest_path_nodes(board, self.player_id)
        opp_path = self.get_shortest_path_nodes(board, self.opponent_id)
        intersection = my_path.intersection(opp_path)

        def move_score(cell):
            score = 0
            # Prioridad 1: Casillas de intersección (Doble valor: avanzo y bloqueo)
            if cell in intersection: score -= 2000
            # Prioridad 2: ¡MI CAMINO! (Ataque) -> Mucho más peso que antes
            elif cell in my_path: score -= 1500
            # Prioridad 3: Bloquear el camino del oponente
            elif cell in opp_path: score -= 500
            
            # Desempate: Cercanía al centro
            score += abs(cell[0] - size//2) + abs(cell[1] - size//2)
            return score

        empty_cells.sort(key=move_score)

        best_score = -math.inf
        best_move = empty_cells[0]
        alpha = -math.inf
        beta = math.inf

        # 4. BÚSQUEDA MINIMAX
        for r, c in empty_cells:
            board.board[r][c] = self.player_id
            score = self.minimax(board, current_depth - 1, alpha, beta, False)
            board.board[r][c] = 0

            if score > best_score:
                best_score = score
                best_move = (r, c)
            
            alpha = max(alpha, best_score)
            if beta <= alpha: break

        return best_move

    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        # Estados terminales absolutos
        if board.check_connection(self.player_id): return 10000 + depth
        if board.check_connection(self.opponent_id): return -10000 - depth

        if depth == 0:
            return self.evaluate_board(board)

        empty_cells = self.get_empty_cells(board)
        if not empty_cells: return 0

        if is_maximizing:
            max_eval = -math.inf
            for r, c in empty_cells:
                board.board[r][c] = self.player_id
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.board[r][c] = 0
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = math.inf
            for r, c in empty_cells:
                board.board[r][c] = self.opponent_id
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.board[r][c] = 0
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval

    def evaluate_board(self, board: HexBoard) -> float:
        my_dist = self.shortest_path_value(board, self.player_id)
        opp_dist = self.shortest_path_value(board, self.opponent_id)
        
        if opp_dist == 0: return -10000
        if my_dist == 0: return 10000
        
        score = 0
        # Penalización extrema si el oponente está a 1 paso de ganar
        if opp_dist == 1: score -= 3000
        # Bonificación si nosotros estamos a 1 paso
        if my_dist == 1: score += 3000
        
        # Nueva fórmula: Penalizamos MÁS fuertemente nuestra propia distancia (Queremos ganar, no solo empatar/bloquear)
        score += (opp_dist * 10) - (my_dist * 30)
        return score

    def shortest_path_value(self, board: HexBoard, player: int) -> int:
        size = board.size
        queue = deque()
        visited = {} 

        if player == 1: 
            for r in range(size):
                if board.board[r][0] == 2: continue
                cost = 0 if board.board[r][0] == 1 else 1
                queue.append((r, 0, cost))
                visited[(r, 0)] = cost
        else: 
            for c in range(size):
                if board.board[0][c] == 1: continue
                cost = 0 if board.board[0][c] == 2 else 1
                queue.append((0, c, cost))
                visited[(0, c)] = cost

        min_cost = 1000

        while queue:
            r, c, cost = queue.popleft()
            if cost >= min_cost: continue

            if (player == 1 and c == size - 1) or (player == 2 and r == size - 1):
                min_cost = min(min_cost, cost)
                continue

            for nr, nc in self.get_neighbors(r, c, size):
                if board.board[nr][nc] == (2 if player == 1 else 1): continue
                new_cost = cost + (0 if board.board[nr][nc] == player else 1)
                
                if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                    visited[(nr, nc)] = new_cost
                    if board.board[nr][nc] == player: queue.appendleft((nr, nc, new_cost))
                    else: queue.append((nr, nc, new_cost))
        
        return min_cost

    def get_shortest_path_nodes(self, board: HexBoard, player: int) -> set:
        size = board.size
        queue = deque()
        visited = {} 

        if player == 1:
            for r in range(size):
                if board.board[r][0] == 2: continue
                cost = 0 if board.board[r][0] == 1 else 1
                queue.append((r, 0, cost, {(r, 0)}))
                visited[(r, 0)] = cost
        else:
            for c in range(size):
                if board.board[0][c] == 1: continue
                cost = 0 if board.board[0][c] == 2 else 1
                queue.append((0, c, cost, {(0, c)}))
                visited[(0, c)] = cost

        min_cost = 1000
        best_path = set()

        while queue:
            r, c, cost, path = queue.popleft()
            if cost > min_cost: continue

            if (player == 1 and c == size - 1) or (player == 2 and r == size - 1):
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
                continue

            for nr, nc in self.get_neighbors(r, c, size):
                if board.board[nr][nc] == (2 if player == 1 else 1): continue
                new_cost = cost + (0 if board.board[nr][nc] == player else 1)
                
                if (nr, nc) not in visited or new_cost < visited[(nr, nc)]:
                    visited[(nr, nc)] = new_cost
                    new_path = path | {(nr, nc)}
                    if board.board[nr][nc] == player: queue.appendleft((nr, nc, new_cost, new_path))
                    else: queue.append((nr, nc, new_cost, new_path))
        
        return best_path

    def get_neighbors(self, r: int, c: int, size: int) -> list:
        if r % 2 == 0:
            directions = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
        else:
            directions = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
        res = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                res.append((nr, nc))
        return res

    def get_empty_cells(self, board: HexBoard) -> list:
        return [(r, c) for r in range(board.size) for c in range(board.size) if board.board[r][c] == 0]