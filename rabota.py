from fractions import Fraction
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set


@dataclass(frozen=True)
class CanonicalTerm:
    coeff: Fraction
    terms: Tuple[str, ...]
    iterations: Tuple["CanonicalForm", ...]

    def __hash__(self):
        return hash((self.coeff, self.terms, self.iterations))


@dataclass(frozen=True)
class CanonicalForm:
    terms: Tuple[CanonicalTerm, ...]

    def normalize(self) -> "CanonicalForm":
        merged: Dict[Tuple[Tuple[str, ...], Tuple["CanonicalForm", ...]], Fraction] = {}
        total = sum(t.coeff for t in self.terms)

        for t in self.terms:
            key = (t.terms, t.iterations)
            merged[key] = merged.get(key, Fraction(0)) + t.coeff

        if total == 0:
            return CanonicalForm(())

        new_terms = [
            CanonicalTerm(coeff=w / total, terms=key[0], iterations=key[1])
            for key, w in merged.items()
            if w != 0
        ]
        return CanonicalForm(tuple(sorted(new_terms, key=lambda x: (x.terms, x.iterations))))

    def __hash__(self):
        return hash(self.terms)

    def __eq__(self, other):
        return self.terms == other.terms


@dataclass
class ASTNode:
    kind: str
    coeff: Fraction = Fraction(1)
    children: List["ASTNode"] = None
    symbol: str = ''

    def __post_init__(self):
        self.children = self.children or []


class RegexParser:
    def __init__(self, expr: str):
        self.expr = expr.replace('∪', '|')
        self.pos = 0

    def _peek(self) -> str:
        return self.expr[self.pos] if self.pos < len(self.expr) else ''

    def _consume(self) -> str:
        ch = self._peek()
        self.pos += 1
        return ch

    def parse(self) -> ASTNode:
        node = self._parse_union()
        if self.pos < len(self.expr):
            raise ValueError(f"Unexpected tail: {self.expr[self.pos:]}")
        return node

    def _parse_union(self):
        left = self._parse_concat()
        while self._peek() == '|':
            self._consume()
            left = ASTNode('union', children=[left, self._parse_concat()])
        return left

    def _parse_concat(self):
        left = self._parse_star()
        while True:
            nxt = self._peek()
            if nxt in ')|' or nxt == '':
                break
            left = ASTNode('concat', children=[left, self._parse_star()])
        return left

    def _parse_star(self):
        node = self._parse_primary()
        while self._peek() == '*':
            self._consume()
            node = ASTNode('star', children=[node])
        return node

    def _parse_primary(self):
        ch = self._peek()
        if ch == '(':
            self._consume()
            node = self._parse_union()
            if self._peek() != ')':
                raise ValueError("missing ')'")
            self._consume()
            return node

        if ch.isdigit() or ch == '.':
            coeff = self._parse_number()
            child = self._parse_primary()
            return ASTNode('coeff', coeff=coeff, children=[child])

        if ch == 'x':
            return ASTNode('term', symbol=self._parse_term())

        raise ValueError(f'unexpected char {ch!r} at {self.pos}')

    def _parse_number(self) -> Fraction:
        s = ''
        while self._peek().isdigit() or self._peek() == '.':
            s += self._consume()
        return Fraction(s).limit_denominator()

    def _parse_term(self) -> str:
        s = ''
        if self._peek() == 'x':
            s += self._consume()
            while self._peek().isdigit():
                s += self._consume()
        return s


def ast_to_canonical(node: ASTNode) -> CanonicalForm:
    k = node.kind
    if k == 'term':
        return CanonicalForm((CanonicalTerm(node.coeff, (node.symbol,), ()),))

    if k == 'coeff':
        inner = ast_to_canonical(node.children[0])
        return CanonicalForm(tuple(
            CanonicalTerm(t.coeff * node.coeff, t.terms, t.iterations)
            for t in inner.terms
        )).normalize()

    if k == 'union':
        left = ast_to_canonical(node.children[0])
        right = ast_to_canonical(node.children[1])
        return CanonicalForm(left.terms + right.terms).normalize()

    if k == 'concat':
        left = ast_to_canonical(node.children[0])
        right = ast_to_canonical(node.children[1])
        new_terms = []
        for lt in left.terms:
            for rt in right.terms:
                new_terms.append(CanonicalTerm(
                    coeff=lt.coeff * rt.coeff,
                    terms=lt.terms + rt.terms,
                    iterations=lt.iterations + rt.iterations
                ))
        return CanonicalForm(tuple(new_terms)).normalize()

    if k == 'star':
        child = ast_to_canonical(node.children[0])
        return CanonicalForm((
            CanonicalTerm(Fraction(1), (), (child,)),
        )).normalize()

    raise ValueError(f"unknown AST node kind {k}")


class AutomatonBuilder:

    def __init__(self):
        self.states: List[CanonicalForm] = []
        self.id_of: Dict[int, int] = {}
        self.tr: Dict[Tuple[int, str], List[Tuple[int, Fraction]]] = {}
        self._cache: Dict[Tuple[int, str], Dict[CanonicalForm, Fraction]] = {}

    def build(self, initial: CanonicalForm, alphabet: Set[str]):
        initial = initial.normalize()
        self.states = [initial]
        self.id_of = {hash(initial): 0}
        self.tr = {}

        queue = [initial]
        while queue:
            cur_cf = queue.pop(0)
            cur_id = self.id_of[hash(cur_cf)]

            for sym in alphabet:
                dist = self._derive_distribution(cur_cf, sym)

                if not dist:
                    continue

                for dest_cf, w in dist.items():
                    dest_cf = dest_cf.normalize()
                    if hash(dest_cf) not in self.id_of:
                        new_id = len(self.states)
                        self.states.append(dest_cf)
                        self.id_of[hash(dest_cf)] = new_id
                        queue.append(dest_cf)
                    else:
                        new_id = self.id_of[hash(dest_cf)]

                    self.tr.setdefault((cur_id, sym), []).append((new_id, w))

        for key, lst in self.tr.items():
            total = sum(w for _, w in lst)
            if total == 0:
                continue
            self.tr[key] = [(to, w / total) for to, w in lst]

    def print_transitions(self):
        n = len(self.states)
        syms = sorted({s for (_, s) in self.tr})

        print("States:")
        for i, st in enumerate(self.states):
            s = " + ".join(f"{float(t.coeff):.4f}·[{','.join(t.terms)}]" if t.terms else f"{float(t.coeff):.4f}·ε"
                           for t in st.terms)
            print(f"a{i}: {s}")
        print()

        for sym in syms:
            print(f"Matrix P({sym}):")
            header = "        " + "".join(f"{f'a{j}':>10}" for j in range(n))
            print(header)
            for i in range(n):
                row = [self._prob(i, sym, j) for j in range(n)]
                print(f"a{i:<2} | " + "".join(f"{float(p):>10.4f}" for p in row))
            print()

    def _prob(self, i: int, sym: str, j: int) -> int:
        return sum(p for to, p in self.tr.get((i, sym), []) if to == j)

    def _derive_distribution(self, cf: CanonicalForm, sym: str) -> Dict[CanonicalForm, Fraction]:
        key = (hash(cf), sym)
        if key in self._cache:
            return self._cache[key]

        dist: Dict[CanonicalForm, Fraction] = {}

        for t in cf.terms:
            if t.terms and t.terms[0] == sym:
                new_term = CanonicalTerm(Fraction(1), t.terms[1:], t.iterations)
                dest = CanonicalForm((new_term,))
                dist[dest] = dist.get(dest, Fraction(0)) + t.coeff

            for inner_cf in t.iterations:
                inner_dist = self._derive_distribution(inner_cf, sym)
                for dest_cf, w in inner_dist.items():
                    dist[dest_cf] = dist.get(dest_cf, Fraction(0)) + t.coeff * w

        self._cache[key] = dist
        return dist


if __name__ == "__main__":
    regex = "(0.003x1(0.09x1(0.4x1|0.1x2)*x1|0.2x1)*x2(0.3x1|0.7x2)*x1|0.21x1(0.3x1|0.6x2)*x1x2)*"
    parser = RegexParser(regex)
    ast = parser.parse()
    initial_cf = ast_to_canonical(ast).normalize()
    builder = AutomatonBuilder()
    builder.build(initial_cf, {'x1', 'x2'})
    builder.print_transitions()
