import random
import copy
import numpy as np
import argparse
import sys

try:
    from backend.fitness_function import (
        mean_absolute_error_fitness,
        mse_fitness,
        rmse_fitness,
        pearson_fitness,
        spearman_fitness,
    )
except ModuleNotFoundError:
    from fitness_function import (
        mean_absolute_error_fitness,
        mse_fitness,
        rmse_fitness,
        pearson_fitness,
        spearman_fitness,
    )

try:
    from backend.stock_data import get_stock_data
except ModuleNotFoundError:
    from stock_data import get_stock_data

def parse_alpha_expression(expr_str):
    """
    Parse a string representation of an alpha expression into an AlphaExpression object.
    
    Examples:
    - "close" -> AlphaExpression('close')
    - "0.5" -> AlphaExpression('const', value=0.5)
    - "(close - open)" -> AlphaExpression('-', left=close_expr, right=open_expr)
    """
    expr_str = expr_str.strip()
    
    # Check if it's a basic variable
    if expr_str in ['high', 'low', 'open', 'close', 'volume']:
        return AlphaExpression(expr_str)
    
    # Check if it's a constant (number)
    try:
        value = float(expr_str)
        return AlphaExpression('const', value=value)
    except ValueError:
        pass

    # Check if it's a unary function call (e.g., log(x))
    if expr_str.startswith('log(') and expr_str.endswith(')'):
        inner_expr = expr_str[4:-1].strip()
        return AlphaExpression('log', left=parse_alpha_expression(inner_expr))
    
    # Check if it's a parenthesized binary expression
    if expr_str.startswith('(') and expr_str.endswith(')'):
        inner_expr = expr_str[1:-1]  # Remove outer parentheses
        
        # Find the main operator (not inside nested parentheses)
        paren_count = 0
        operator_pos = -1
        operator = None
        
        # Look for operators from right to left to handle precedence correctly
        for i in range(len(inner_expr) - 1, -1, -1):
            char = inner_expr[i]
            if char == ')':
                paren_count += 1
            elif char == '(':
                paren_count -= 1
            elif paren_count == 0 and char in ['+', '-', '*', '/']:
                operator_pos = i
                operator = char
                break
        
        if operator_pos != -1:
            left_expr = inner_expr[:operator_pos].strip()
            right_expr = inner_expr[operator_pos + 1:].strip()
            
            return AlphaExpression(
                operator,
                left=parse_alpha_expression(left_expr),
                right=parse_alpha_expression(right_expr)
            )
    
    # If we can't parse it, raise an error
    raise ValueError(f"Cannot parse expression: {expr_str}")

class AlphaExpression:
    """Represents an alpha expression as a tree structure."""
    def __init__(self, op, left=None, right=None, value=None):
        self.op = op  # operation or variable name
        self.left = left
        self.right = right
        self.value = value  # for constants
    
    def evaluate(self, X):
        """Evaluate the expression given stock data X."""
        if self.op in ['high', 'low', 'open', 'close', 'volume']:
            col_map = {'high': 0, 'low': 1, 'open': 2, 'close': 3, 'volume': 4}
            return X[:, col_map[self.op]]
        elif self.op == 'const':
            return np.full(len(X), self.value)
        elif self.op == '+':
            return self.left.evaluate(X) + self.right.evaluate(X)
        elif self.op == '-':
            return self.left.evaluate(X) - self.right.evaluate(X)
        elif self.op == '*':
            return self.left.evaluate(X) * self.right.evaluate(X)
        elif self.op == '/':
            right_val = self.right.evaluate(X)
            return self.left.evaluate(X) / (right_val + 1e-8)  # avoid division by zero
        elif self.op == 'log':
            left_val = self.left.evaluate(X)
            return np.log(np.abs(left_val) + 1e-8)
        elif self.op == 'sign':
            left_val = self.left.evaluate(X)
            return np.sign(left_val)
        else:
            raise ValueError(f"Unknown operation: {self.op}")
    
    def __str__(self):
        """Return string representation of the expression."""
        if self.op in ['high', 'low', 'open', 'close', 'volume']:
            return self.op
        elif self.op == 'const':
            return str(self.value)
        elif self.op in ['+', '-', '*', '/']:
            return f"({self.left} {self.op} {self.right})"
        elif self.op == 'log':
            return f"log({self.left})"
        elif self.op == 'sign':
            return f"sign({self.left})"
        else:
            return str(self.op)
    
    def __call__(self, X):
        """Make it callable like a lambda function."""
        return self.evaluate(X)

class GeneticProgramming:
    def __init__(self, fitness_fn, npop, pcrossover, ppoint, params_init, 
                 stock_data, forward_return, alpha_init, max_generations=50):
        self.fitness_fn = fitness_fn
        self.npop = npop
        self.pcrossover = pcrossover
        self.ppoint = ppoint
        self.params_init = params_init
        self.stock_data = stock_data
        self.forward_return = forward_return
        self.alpha_init = alpha_init
        self.max_generations = max_generations
        self.population = []
        self.history = []

    def initialize_population(self):
        # Start with alpha_init as seed, fill rest randomly
        self.population = [self.alpha_init]
        while len(self.population) < self.npop:
            self.population.append(self.random_alpha())

    def random_alpha(self, max_depth=3):
        """Generate a random alpha expression tree."""
        if max_depth == 0:
            # Terminal node
            if random.random() < 0.8:
                return AlphaExpression(random.choice(['high', 'low', 'open', 'close', 'volume']))
            else:
                return AlphaExpression('const', value=random.uniform(-1, 1))
        else:
            # Non-terminal node
            if random.random() < 0.8:  # 80% chance for binary operations
                op = random.choice(['+', '-', '*', '/', 'sign'])
                left = self.random_alpha(max_depth - 1)
                right = self.random_alpha(max_depth - 1)
                return AlphaExpression(op, left, right)
            else:  # 20% chance for unary operations
                op = 'log'
                left = self.random_alpha(max_depth - 1)
                return AlphaExpression(op, left)

    def evaluate_population(self, population):
        scores = []
        for alpha in population:
            try:
                predictions = alpha(self.stock_data)
                score = self.fitness_fn(predictions, self.forward_return)
            except Exception:
                score = -np.inf  # invalid alpha gets worst score
            scores.append(score)
        return np.array(scores)

    def tournament(self, population, scores, k=3):
        """Select one alpha via tournament selection."""
        idxs = np.random.choice(len(population), k, replace=False)
        best_idx = idxs[np.argmax(scores[idxs])]
        return copy.deepcopy(population[best_idx])

    def crossover(self, alpha1, alpha2):
        """Tree crossover: swap random subtrees."""
        # Simple implementation: choose random parent
        return random.choice([alpha1, alpha2])

    def point_mutation(self, alpha):
        """Replace with a random alpha."""
        return self.random_alpha()

    def evolve(self):
        self.initialize_population()
        scores = self.evaluate_population(self.population)
        best_idx = np.argmax(scores)
        best_alpha = self.population[best_idx]
        best_score = scores[best_idx]

        for t in range(self.max_generations):
            new_population = [best_alpha]  # elitism

            while len(new_population) < self.npop:
                if t + 1 == 1:
                    mutation = "point"
                else:
                    mutation = random.choices(
                        ["crossover", "point", "clone"],
                        weights=[self.pcrossover, self.ppoint, 1 - self.pcrossover - self.ppoint],
                        k=1
                    )[0]

                if mutation == "crossover":
                    p1 = self.tournament(self.population, scores)
                    p2 = self.tournament(self.population, scores)
                    offspring = self.crossover(p1, p2)
                elif mutation == "point":
                    p1 = self.tournament(self.population, scores)
                    offspring = self.point_mutation(p1)
                else:
                    p1 = self.tournament(self.population, scores)
                    offspring = copy.deepcopy(p1)

                if offspring not in new_population:
                    new_population.append(offspring)

            scores = self.evaluate_population(new_population)
            best_idx = np.argmax(scores)
            best_alpha = new_population[best_idx]
            best_score = scores[best_idx]
            self.population = new_population
            self.history.append(best_score)

            print(f"Gen {t+1}: Best fitness = {best_score:.4f}")

        return best_alpha, best_score, self.history


def _load_training_data(ticker: str, years: int) -> tuple[np.ndarray, np.ndarray]:
    df = get_stock_data(ticker, years)
    if hasattr(df, "columns") and isinstance(df.columns, np.ndarray) is False:
        if hasattr(df.columns, "levels"):
            df.columns = [col[0] for col in df.columns]

    close_col = "Close" if "Close" in df.columns else "Adj Close"
    needed = ["High", "Low", "Open", close_col, "Volume"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for GP data: {missing}")

    X = np.column_stack(
        [
            df["High"].to_numpy(),
            df["Low"].to_numpy(),
            df["Open"].to_numpy(),
            df[close_col].to_numpy(),
            df["Volume"].to_numpy(),
        ]
    )
    forward_return = df[close_col].pct_change().shift(-1).fillna(0.0).to_numpy()
    if len(forward_return) > 1:
        X = X[:-1]
        forward_return = forward_return[:-1]

    return X, forward_return


def _resolve_fitness_fn(name: str):
    mapping = {
        "mean_absolute_error_fitness": mean_absolute_error_fitness,
        "mse_fitness": mse_fitness,
        "rmse_fitness": rmse_fitness,
        "pearson_fitness": pearson_fitness,
        "spearman_fitness": spearman_fitness,
    }
    return mapping.get(name, pearson_fitness)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run genetic programming")
    parser.add_argument("--message", default="", help="User message (optional)")
    parser.add_argument("--alpha-expression", required=True, help="Initial alpha expression")
    parser.add_argument("--fitness-function", default="pearson_fitness", help="Fitness function name")
    parser.add_argument("--npop", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--crossover", type=float, default=0.4, help="Crossover rate")
    parser.add_argument("--mutation", type=float, default=0.4, help="Mutation rate")
    parser.add_argument("--ticker", default="^GSPC", help="Market index ticker")
    parser.add_argument("--years", type=int, default=10, help="History length in years")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    try:
        alpha_init = parse_alpha_expression(args.alpha_expression)
        X, forward_return = _load_training_data(args.ticker, args.years)
        fitness_fn = _resolve_fitness_fn(args.fitness_function)
        gp = GeneticProgramming(
            fitness_fn=fitness_fn,
            npop=args.npop,
            pcrossover=args.crossover,
            ppoint=args.mutation,
            params_init={},
            stock_data=X,
            forward_return=forward_return,
            alpha_init=alpha_init,
            max_generations=args.generations,
        )
        best_alpha, best_score, _history = gp.evolve()
        print(f"Best Alpha: {best_alpha}")
        print(f"Best Fitness: {best_score}")
        return 0
    except Exception as exc:
        print(f"GP run failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

