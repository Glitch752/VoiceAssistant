from enum import Enum
import math
from typing import Callable

class TokenType(Enum):
    NUMBER = 1
    PLUS = 2
    MINUS = 3
    MULTIPLY = 4
    DIVIDE = 5
    EXPONENT = 6
    LPAREN = 7
    RPAREN = 8
    IDENTIFIER = 9

class Token:
    def __init__(self, type: TokenType, value: str):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

def tokenize(expression: str) -> list[Token]:
    """
    Tokenize the input expression into a list of tokens.
    
    Each token is represented as a tuple of (TokenType, value).
    """
    
    tokens: list[Token] = []
    i = 0
    while i < len(expression):
        char = expression[i]

        if char.isdigit() or (char == '.' and i + 1 < len(expression) and expression[i + 1].isdigit()):
            start = i
            while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                i += 1
            tokens.append(Token(TokenType.NUMBER, expression[start:i]))
        elif char == '+':
            tokens.append(Token(TokenType.PLUS, char))
            i += 1
        elif char == '-':
            tokens.append(Token(TokenType.MINUS, char))
            i += 1
        elif char == '*':
            tokens.append(Token(TokenType.MULTIPLY, char))
            i += 1
        elif char == '/':
            tokens.append(Token(TokenType.DIVIDE, char))
            i += 1
        elif char == '^':
            tokens.append(Token(TokenType.EXPONENT, char))
            i += 1
        elif char == '(':
            tokens.append(Token(TokenType.LPAREN, char))
            i += 1
        elif char == ')':
            tokens.append(Token(TokenType.RPAREN, char))
            i += 1
        elif char.isalpha():
            start = i
            while i < len(expression) and expression[i].isalpha():
                i += 1
            tokens.append(Token(TokenType.IDENTIFIER, expression[start:i]))
        elif char.isspace():
            i += 1
        else:
            raise ValueError(f"Unexpected character: {char}")
    return tokens

def eval_expression(tokens: list[Token]) -> float:
    """
    Evaluate the expression represented by the list of tokens.
    Because our expression language is so simple, we don't need a full parser.
    We just use a single-pass recursive descent parser.
    """
    return eval_add_sub(tokens)

def eval_add_sub(tokens: list[Token]) -> float:
    """
    Evaluate addition and subtraction.
    """
    result = eval_mul_div(tokens)
    while tokens and (tokens[0].type == TokenType.PLUS or tokens[0].type == TokenType.MINUS):
        op = tokens.pop(0)
        if op.type == TokenType.PLUS:
            result += eval_mul_div(tokens)
        elif op.type == TokenType.MINUS:
            result -= eval_mul_div(tokens)
    return result

def eval_mul_div(tokens: list[Token]) -> float:
    """
    Evaluate multiplication and division.
    """
    result = eval_exponent(tokens)
    while tokens and (tokens[0].type == TokenType.MULTIPLY or tokens[0].type == TokenType.DIVIDE):
        op = tokens.pop(0)
        if op.type == TokenType.MULTIPLY:
            result *= eval_exponent(tokens)
        elif op.type == TokenType.DIVIDE:
            result /= eval_exponent(tokens)
    return result

def eval_exponent(tokens: list[Token]) -> float:
    """
    Evaluate exponentiation.
    """
    result = eval_unary(tokens)
    while tokens and tokens[0].type == TokenType.EXPONENT:
        tokens.pop(0)  # Remove the exponent token
        result **= eval_unary(tokens)
    return result

def eval_unary(tokens: list[Token]) -> float:
    """
    Evaluate unary operations (like negation).
    """
    if tokens and tokens[0].type == TokenType.MINUS:
        tokens.pop(0)  # Remove the '-' token
        return -eval_primary(tokens)
    return eval_primary(tokens)

def eval_primary(tokens: list[Token]) -> float:
    """
    Evaluate primary expressions (numbers, identifiers, and parenthesized expressions).
    """
    if not tokens:
        raise ValueError("Unexpected end of expression")

    token = tokens.pop(0)
    
    if token.type == TokenType.NUMBER:
        return float(token.value)
    elif token.type == TokenType.IDENTIFIER:
        return eval_function(token.value, tokens)
    elif token.type == TokenType.LPAREN:
        result = eval_expression(tokens)
        if tokens and tokens[0].type == TokenType.RPAREN:
            tokens.pop(0)  # Remove the ')'
            return result
        else:
            raise ValueError("Missing closing parenthesis")
    else:
        raise ValueError(f"Unexpected token: {token}")

def eval_function(name: str, tokens: list[Token]) -> float:
    """
    Evaluate functions (like sin, cos, tan, log, sqrt).
    """
    FUNCTIONS: dict[str, float | Callable[[float], float]] = {
        'sin': lambda x: math.sin(x),
        'cos': lambda x: math.cos(x),
        'tan': lambda x: math.tan(x),
        'log': lambda x: math.log(x),
        'sqrt': lambda x: math.sqrt(x),
        # Just throw in a bunch of stuff we don't explicitly say we support
        # ...in case, I guess? Who knows with LLMs.
        'abs': lambda x: abs(x),
        'exp': lambda x: math.exp(x),
        'log10': lambda x: math.log10(x),
        'log2': lambda x: math.log2(x),
        'ln': lambda x: math.log(x),
        'asin': lambda x: math.asin(x),
        'acos': lambda x: math.acos(x),
        'atan': lambda x: math.atan(x),
        'arcsin': lambda x: math.asin(x),
        'arccos': lambda x: math.acos(x),
        'arctan': lambda x: math.atan(x),
        
        'e': math.e,
        'pi': math.pi,
        'tau': math.tau,
        'phi': (1 + math.sqrt(5)) / 2
    }
    
    if name in FUNCTIONS:
        func = FUNCTIONS[name]
        # If the function is a lambda, expect a call
        if callable(func):
            if tokens and tokens[0].type == TokenType.LPAREN:
                tokens.pop(0)  # Remove the '('
                arg = eval_expression(tokens)
                if tokens and tokens[0].type == TokenType.RPAREN:
                    tokens.pop(0)  # Remove the ')'
                else:
                    raise ValueError("Missing closing parenthesis")
                return func(arg)
            raise ValueError(f"Missing argument for function: {name}")
        # If it's a constant, return its value
        return func
    else:
        raise ValueError(f"Unknown function or variable: {name}")

def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result as a string.
    
    The calculator supports addition (+), subtraction (-), multiplication (*), division (/), exponentiation (^), trigonometric functions (sin, cos, tan),
    logarithms (log), square roots (sqrt), and mathematical constants (e, pi). Standard textual math notation is used."
    """
    
    try:
        tokens = tokenize(expression)
        return str(eval_expression(tokens))
    except Exception as e:
        return f"Error: {str(e)}"