from fractions import Fraction
from collections import defaultdict


class ASTNode:
    def __init__(self, node_type, symbol=None, coeff=None, children=None):
        self.node_type = node_type
        self.symbol = symbol
        self.coeff = coeff
        self.children = children or []

    def __repr__(self):
        return f"ASTNode({self.node_type}, {self.symbol}, {self.coeff})"


class State:
    def __init__(self, name, language):
        self.name = name
        self.language = language


class StochasticAutomaton:
    def __init__(self, alphabet):
        self.states = []
        self.alphabet = alphabet
        self.initial_dist = []
        self.transitions = defaultdict(list)
        self.output_vector = []

    def add_transition(self, from_state, symbol, to_state, weight):
        self.transitions[(from_state, symbol)].append((to_state, float(weight)))


class RegexParser:
    def __init__(self, expr: str):
        self.expr = expr.replace('∪', '|').replace(' ', '')
        self.pos = 0

    def _peek(self):
        return self.expr[self.pos] if self.pos < len(self.expr) else ''

    def _consume(self):
        ch = self._peek()
        if ch: self.pos += 1
        return ch

    def parse(self):
        node = self._parse_union()
        if self.pos < len(self.expr):
            raise ValueError(f"Unexpected tail: {self.expr[self.pos:]}")
        return node

    def _parse_union(self):
        nodes = [self._parse_concat()]
        while self._peek() == '|':
            self._consume()
            nodes.append(self._parse_concat())
        return ASTNode('union', children=nodes) if len(nodes) > 1 else nodes[0]

    def _parse_concat(self):
        nodes = [self._parse_star()]
        while self._peek() not in ('|', ')', '*', ''):
            nodes.append(self._parse_star())
        return ASTNode('concat', children=nodes) if len(nodes) > 1 else nodes[0]

    def _parse_star(self):
        node = self._parse_primary()
        while self._peek() == '*':
            self._consume()
            node = ASTNode('star', children=[node])
        return node

    def _parse_primary(self):
        if self._peek() == '(':
            self._consume()
            node = self._parse_union()
            if self._peek() != ')':
                raise ValueError("Missing ')'")
            self._consume()
            return node

        if self._peek().isdigit() or self._peek() in ('.',):
            coeff = self._parse_number()
            child = self._parse_primary()
            return ASTNode('coeff', coeff=coeff, children=[child])

        if self._peek() == 'x':
            symbol = self._parse_symbol()
            return ASTNode('term', symbol=symbol)

        raise ValueError(f"Unexpected char '{self._peek()}' at pos {self.pos}")

    def _parse_number(self):
        s = []
        while self._peek().isdigit() or self._peek() == '.':
            s.append(self._consume())
        return Fraction(''.join(s)).limit_denominator()

    def _parse_symbol(self):
        self._consume()  # Пропускаем 'x'
        num = []
        while self._peek().isdigit():
            num.append(self._consume())
        return f"x{''.join(num)}"


def compute_left_derivative(node, symbol):
    if node.node_type == 'coeff':
        child_deriv = compute_left_derivative(node.children[0], symbol)
        return ASTNode('coeff', coeff=node.coeff, children=[child_deriv])

    elif node.node_type == 'term':
        if node.symbol == symbol:
            return ASTNode('coeff', coeff=Fraction(1))
        else:
            return ASTNode('coeff', coeff=Fraction(0))

    elif node.node_type == 'concat':
        left_deriv = compute_left_derivative(node.children[0], symbol)
        right_part = node.children[1]
        empty_weight = compute_empty_weight(node.children[0])

        deriv = ASTNode('union', children=[
            ASTNode('concat', children=[left_deriv, right_part]),
            ASTNode('coeff', coeff=empty_weight, children=[
                compute_left_derivative(node.children[1], symbol)
            ])
        ])
        return deriv

    elif node.node_type == 'union':
        return ASTNode('union', children=[
            compute_left_derivative(child, symbol) for child in node.children
        ])

    elif node.node_type == 'star':
        inner_deriv = compute_left_derivative(node.children[0], symbol)
        return ASTNode('concat', children=[
            inner_deriv,
            ASTNode('star', children=node.children)
        ])

    return ASTNode('coeff', coeff=Fraction(0))


def compute_empty_weight(node):
    if node.node_type == 'coeff':
        return node.coeff * compute_empty_weight(node.children[0])

    elif node.node_type == 'term':
        return Fraction(0)

    elif node.node_type == 'concat':
        return compute_empty_weight(node.children[0]) * compute_empty_weight(node.children[1])

    elif node.node_type == 'union':
        return sum(compute_empty_weight(child) for child in node.children)

    elif node.node_type == 'star':
        return Fraction(1)

    return Fraction(0)


def build_stochastic_automaton(expr, alphabet):
    # Парсинг входного выражения
    parser = RegexParser(expr)
    ast = parser.parse()

    # Инициализация автомата
    automaton = StochasticAutomaton(alphabet)
    initial_state = State("a0", ast)
    automaton.states.append(initial_state)
    automaton.initial_dist = [1.0]

    # Основной цикл обработки состояний
    queue = [initial_state]
    processed = set()

    while queue:
        state = queue.pop(0)
        if state.name in processed:
            continue
        processed.add(state.name)

        # Вычисляем производные для всех символов алфавита
        for symbol in alphabet:
            deriv_ast = compute_left_derivative(state.language, symbol)
            simplified_ast = simplify_ast(deriv_ast)

            # Поиск существующего состояния
            existing_state = next(
                (s for s in automaton.states if ast_equals(s.language, simplified_ast)),
                None
            )

            weight = compute_empty_weight(simplified_ast)

            if existing_state:
                automaton.add_transition(state.name, symbol, existing_state.name, weight)
            else:
                new_state = State(f"a{len(automaton.states)}", simplified_ast)
                automaton.states.append(new_state)
                automaton.add_transition(state.name, symbol, new_state.name, weight)
                queue.append(new_state)

    # Нормализация весов
    normalize_weights(automaton)

    # Построение выходного вектора
    automaton.output_vector = [compute_empty_weight(s.language) for s in automaton.states]

    return automaton


def simplify_ast(node):
    if node.node_type == 'coeff' and node.coeff == 0:
        return ASTNode('coeff', coeff=Fraction(0))

    if node.node_type == 'union':
        simplified_children = []
        for child in node.children:
            schild = simplify_ast(child)
            if schild.node_type == 'union':
                simplified_children.extend(schild.children)
            elif schild.coeff != 0:  # type: ignore
                simplified_children.append(schild)
        if not simplified_children:
            return ASTNode('coeff', coeff=Fraction(0))
        return ASTNode('union', children=simplified_children)

    if node.node_type == 'concat':
        left = simplify_ast(node.children[0])
        right = simplify_ast(node.children[1])
        if left.coeff == 0 or right.coeff == 0:  # type: ignore
            return ASTNode('coeff', coeff=Fraction(0))
        return ASTNode('concat', children=[left, right])

    return node


def ast_equals(a, b):
    if a.node_type != b.node_type:
        return False
    if a.coeff != b.coeff:
        return False
    if a.symbol != b.symbol:
        return False
    return all(ast_equals(c1, c2) for c1, c2 in zip(a.children, b.children))


def normalize_weights(automaton):
    for (from_state, symbol), transitions in automaton.transitions.items():
        total = sum(weight for _, weight in transitions)
        if total > 1e-6:
            normalized = [(to, weight / total) for to, weight in transitions]
            automaton.transitions[(from_state, symbol)] = normalized
        else:
            automaton.transitions[(from_state, symbol)] = []


# Пример использования
if __name__ == "__main__":
    expr = "0.5x1(0.1x1|0.2x2)*0.9x3"
    alphabet = ["x1", "x2", "x3"]

    automaton = build_stochastic_automaton(expr, alphabet)

    print("Состояния:")
    for state in automaton.states:
        print(f"  {state.name}: {state.language}")

    print("\nПереходы:")
    for (from_state, symbol), transitions in automaton.transitions.items():
        for to, weight in transitions:
            print(f"  {from_state} --{symbol}-> {to} ({weight:.2f})")

    print("\nВыходной вектор:", [f"{v:.2f}" for v in automaton.output_vector])