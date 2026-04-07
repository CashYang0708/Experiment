import numpy as np
import pandas as pd


class AlphaGP():
    def __init__(self, df: pd.DataFrame, alpha_expression: str):
        self.df_data = df.copy()
        self.alpha_expression = alpha_expression
        self.open = df['Open']
        self.high = df['High']
        self.low = df['Low']
        self.close = df['Adj Close'] 
        self.volume = df['Volume']
        self.df_data['weights'] = self.calculate()
        

    def calculate(self) -> pd.Series:
        """Calculate the alpha values based on the provided expression."""
        # Create evaluation context with all necessary variables and functions
        eval_context = {
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'sign': np.sign,
            'log': lambda x: np.log(np.abs(x) + 1e-8),
        }
        
        return eval(self.alpha_expression, {"__builtins__": {}}, eval_context)