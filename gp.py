import random
import copy
import numpy as np
import argparse
import json

try:
    import fitness_function as external_fitness
except ImportError:
    external_fitness = None

try:
    from backtest.executor import GpTemplate
except Exception:
    GpTemplate = None

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
        elif self.op == 'sign':
            return np.sign(self.left.evaluate(X))
        elif self.op == 'log':
            return np.log(np.abs(self.left.evaluate(X)) + 1e-8)
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
        elif self.op == 'sign':
            return f"sign({self.left})"
        elif self.op == 'log':
            return f"log({self.left})"
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
            if random.random() < 0.7:  # 70% chance for binary operations
                op = random.choice(['+', '-', '*', '/'])
                left = self.random_alpha(max_depth - 1)
                right = self.random_alpha(max_depth - 1)
                return AlphaExpression(op, left, right)
            else:  # 30% chance for unary operations
                op = random.choice(['log', 'sign'])
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


def correlation_fitness(predictions, forward_return):
    """Simple fitness by correlation between alpha signal and forward return."""
    p = np.asarray(predictions)
    r = np.asarray(forward_return)
    if len(p) != len(r) or len(p) < 2:
        return -np.inf
    if np.std(p) < 1e-12 or np.std(r) < 1e-12:
        return -np.inf
    corr = np.corrcoef(p, r)[0, 1]
    if np.isnan(corr):
        return -np.inf
    return float(corr)


def get_fitness_function(name):
    """Resolve fitness function by name from local or external registry."""
    local_registry = {
        "correlation_fitness": correlation_fitness,
    }

    if name in local_registry:
        return local_registry[name]

    if external_fitness is not None and hasattr(external_fitness, name):
        fn = getattr(external_fitness, name)
        if callable(fn):
            return fn

    available = list(local_registry.keys())
    if external_fitness is not None:
        available.extend(
            [
                n
                for n in dir(external_fitness)
                if n.endswith("_fitness") and callable(getattr(external_fitness, n))
            ]
        )
    available = sorted(set(available))
    raise ValueError(f"Unknown fitness function: {name}. Available: {available}")


def run_backtest_for_alpha(alpha_expression):
    """Run backtest for best alpha expression via backtest/executor.py."""
    if GpTemplate is None:
        return {
            "status": "skipped",
            "reason": "backtest.executor.GpTemplate import failed",
        }

    try:
        template = GpTemplate(alpha_expression)
        result = template.run()
        return {
            "status": "ok",
            "metrics": result,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
        }


def run_gp_from_query(
    query,
    npop=20,
    generations=5,
    seed=42,
    pcrossover=0.4,
    ppoint=0.4,
    fitness_name="pearson_fitness",
    run_backtest=False,
):
    """Run a lightweight GP search and return summary text."""
    if pcrossover < 0 or ppoint < 0 or pcrossover + ppoint > 1:
        raise ValueError("pcrossover and ppoint must be >=0 and pcrossover + ppoint <= 1")

    random.seed(seed)
    np.random.seed(seed)

    n = 240
    # Synthetic market-like matrix: [high, low, open, close, volume]
    base = np.random.normal(0, 1, (n, 5))
    base[:, 0] = np.abs(base[:, 0]) + 100  # high
    base[:, 1] = base[:, 0] - np.abs(np.random.normal(0, 0.5, n))  # low
    base[:, 2] = base[:, 1] + np.abs(np.random.normal(0, 0.3, n))  # open
    base[:, 3] = base[:, 1] + np.abs(np.random.normal(0, 0.6, n))  # close
    base[:, 4] = np.abs(np.random.normal(1e6, 2e5, n))  # volume

    forward_return = np.random.normal(0, 0.02, n)
    alpha_init = parse_alpha_expression("close")

    fitness_fn = get_fitness_function(fitness_name)

    gp = GeneticProgramming(
        fitness_fn=fitness_fn,
        npop=npop,
        pcrossover=pcrossover,
        ppoint=ppoint,
        params_init={},
        stock_data=base,
        forward_return=forward_return,
        alpha_init=alpha_init,
        max_generations=generations,
    )

    best_alpha, best_score, history = gp.evolve()
    backtest_text = ""
    if run_backtest:
        backtest_result = run_backtest_for_alpha(str(best_alpha))

        if backtest_result.get("status") == "ok":
            backtest_text = "\nBacktest: ok\n" + json.dumps(backtest_result.get("metrics", {}), ensure_ascii=False)
        else:
            backtest_text = (
                f"\nBacktest: {backtest_result.get('status', 'unknown')}\n"
                f"Reason: {backtest_result.get('reason', 'N/A')}"
            )

    return (
        f"Query: {query}\n"
        f"Fitness Function: {fitness_fn.__name__}\n"
        f"Crossover: {pcrossover:.3f}\n"
        f"Mutation: {ppoint:.3f}\n"
        f"Best Alpha: {best_alpha}\n"
        f"Best Fitness: {best_score:.6f}\n"
        f"Generations: {generations}\n"
        f"History Length: {len(history)}"
        f"{backtest_text}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run genetic programming alpha search")
    parser.add_argument("--message", default="", help="Original user query text")
    parser.add_argument("--npop", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=5, help="Max generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--crossover", type=float, default=0.4, help="Crossover probability")
    parser.add_argument("--mutation", type=float, default=0.4, help="Point mutation probability")
    parser.add_argument(
        "--fitness-function",
        default="pearson_fitness",
        help="Fitness function name (from fitness_function.py or local), e.g. pearson_fitness",
    )
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        help="Run backtest with backtest/executor.py after finding best alpha",
    )
    args = parser.parse_args()

    result = run_gp_from_query(
        query=args.message,
        npop=args.npop,
        generations=args.generations,
        seed=args.seed,
        pcrossover=args.crossover,
        ppoint=args.mutation,
        fitness_name=args.fitness_function,
        run_backtest=args.run_backtest,
    )
    print(result)


if __name__ == "__main__":
    main()

