from player import Player
from board import HexBoard
from collections import deque
import math

class SmartPlayer(Player):
    def __init__(self, player_id: int, max_depth: int = 2):
        """
        Inicializa la IA.
        max_depth: Profundidad del algoritmo Minimax. 
        (Un valor de 2 o 3 es seguro para tableros medianos en Python).
        """
        super().__init__(player_id)
        self.opponent_id = 2 if player_id == 1 else 1
        self.max_depth = max_depth

    def play(self, board: HexBoard) -> tuple:
        """Decide la mejor jugada evaluando posibles escenarios con Minimax."""
        best_score = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf

        empty_cells = self.get_empty_cells(board)
        
        # Heurística de ordenación: Revisar primero las casillas más centrales 
        # (ayuda a la poda Alfa-Beta a ser más eficiente)
        center = board.size // 2
        empty_cells.sort(key=lambda x: abs(x[0] - center) + abs(x[1] - center))

        for r, c in empty_cells:
            # 1. Simular la jugada
            board.board[r][c] = self.player_id
            
            # 2. Evaluar usando Minimax
            score = self.minimax(board, self.max_depth - 1, alpha, beta, False)
            
            # 3. Deshacer la jugada
            board.board[r][c] = 0

            # 4. Actualizar el mejor movimiento
            if score > best_score:
                best_score = score
                best_move = (r, c)
            
            alpha = max(alpha, best_score)

        # Si por alguna razón no se encontró jugada, devolver la primera vacía
        if best_move is None and empty_cells:
            best_move = empty_cells[0]

        return best_move

    def minimax(self, board: HexBoard, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
        """Algoritmo Minimax con Poda Alfa-Beta."""
        # Verificar estados terminales primero
        if board.check_connection(self.player_id):
            return 10000 + depth  # Prioriza ganar en menos pasos
        if board.check_connection(self.opponent_id):
            return -10000 - depth # Evita que el oponente gane

        # Llegamos al límite de profundidad, usamos la Heurística
        if depth == 0:
            return self.evaluate_board(board)

        empty_cells = self.get_empty_cells(board)
        if not empty_cells:
            return 0 # Tablero lleno (empate, aunque en HEX siempre hay un ganador)

        if is_maximizing:
            max_eval = -math.inf
            for r, c in empty_cells:
                board.board[r][c] = self.player_id
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.board[r][c] = 0
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # Poda Alfa
            return max_eval
        else:
            min_eval = math.inf
            for r, c in empty_cells:
                board.board[r][c] = self.opponent_id
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.board[r][c] = 0
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # Poda Beta
            return min_eval

    def evaluate_board(self, board: HexBoard) -> float:
        """
        Función heurística: Calcula la ventaja contando la cantidad de fichas 
        faltantes para que cada jugador conecte sus lados.
        """
        my_distance = self.shortest_path(board, self.player_id)
        opp_distance = self.shortest_path(board, self.opponent_id)
        
        # Una puntuación alta significa que el oponente está lejos de ganar y nosotros cerca
        return opp_distance - my_distance

    def shortest_path(self, board: HexBoard, player: int) -> int:
        """
        Busca el costo mínimo (fichas faltantes) para conectar los bordes
        usando Búsqueda en Anchura con pesos 0-1 (0-1 BFS).
        """
        size = board.size
        queue = deque()
        distances = {}

        # Inicializar los bordes iniciales
        if player == 1: # Jugador 1 conecta Izquierda (col=0) a Derecha (col=size-1)
            for r in range(size):
                if board.board[r][0] == player:
                    queue.appendleft((r, 0, 0))
                    distances[(r, 0)] = 0
                elif board.board[r][0] == 0:
                    queue.append((r, 0, 1))
                    distances[(r, 0)] = 1
        else: # Jugador 2 conecta Arriba (row=0) a Abajo (row=size-1)
            for c in range(size):
                if board.board[0][c] == player:
                    queue.appendleft((0, c, 0))
                    distances[(0, c)] = 0
                elif board.board[0][c] == 0:
                    queue.append((0, c, 1))
                    distances[(0, c)] = 1

        visited = set()

        while queue:
            r, c, cost = queue.popleft()

            # Condición de victoria / fin de trayecto
            if player == 1 and c == size - 1:
                return cost
            if player == 2 and r == size - 1:
                return cost

            if (r, c) in visited:
                continue
            visited.add((r, c))

            for nr, nc in self.get_neighbors(r, c, size):
                if (nr, nc) in visited: 
                    continue
                
                cell_val = board.board[nr][nc]
                opp = 2 if player == 1 else 1
                
                if cell_val == opp:
                    continue # Camino bloqueado por el oponente

                new_cost = cost + (0 if cell_val == player else 1)
                
                if (nr, nc) not in distances or new_cost < distances[(nr, nc)]:
                    distances[(nr, nc)] = new_cost
                    if cell_val == player:
                        queue.appendleft((nr, nc, new_cost)) # Prioriza nuestras fichas (costo 0)
                    else:
                        queue.append((nr, nc, new_cost))     # Casillas vacías (costo 1)

        return 1000 # Retorna un costo muy alto si está completamente bloqueado

    def get_neighbors(self, r: int, c: int, size: int) -> list:
        """
        Obtiene los vecinos válidos de una celda hexagonal (even-r layout).
        En even-r, las filas pares se desplazan hacia la derecha.
        """
        if r % 2 == 0: # Fila par
            directions = [
                (0, -1), (0, 1),   # Izquierda, Derecha
                (-1, 0), (-1, 1),  # Arriba-Izquierda, Arriba-Derecha
                (1, 0), (1, 1)     # Abajo-Izquierda, Abajo-Derecha
            ]
        else: # Fila impar
            directions = [
                (0, -1), (0, 1),   # Izquierda, Derecha
                (-1, -1), (-1, 0), # Arriba-Izquierda, Arriba-Derecha
                (1, -1), (1, 0)    # Abajo-Izquierda, Abajo-Derecha
            ]

        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbors.append((nr, nc))
        return neighbors

    def get_empty_cells(self, board: HexBoard) -> list:
        """Devuelve una lista con las coordenadas de todas las celdas vacías."""
        empty = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    empty.append((r, c))
        return empty