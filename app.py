"""
Memscope - a visual, educational C-subset interpreter.

HOW THE PIECES FIT TOGETHER
---------------------------
    C source text
        -> tokenize()              (section 6)  text   -> tokens
        -> Parser                  (section 7)  tokens -> AST
        -> Interpreter             (section 8)  AST    -> steps (one Python generator `yield` per step)
        -> ProgramState            (section 3)  the single source of truth (memory, variables, frames, output)
        -> events                  (section 9)  generic facts about what each step did
        -> JSON over /api/execute  (section 12)
        -> browser                 (section 11) draws ProgramState + events; it never runs C itself.

The engine is STATELESS between HTTP requests (Vercel runs this as a serverless function, so a Python
generator cannot survive between two requests). To perform "step N" the server re-parses the source and
replays N steps from the beginning. The engine is deterministic, so this always gives the same result.

IMPORTANT: the simulated machine is a teaching model, NOT real C memory (see section 4).
"""

# ============================================
# 1. IMPORTS
# ============================================
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Generator, Optional

from flask import Flask, Response, jsonify, request

# ============================================
# 2. CONFIGURATION
# ============================================
MEM_BASE = 0x1000      # Address of the first simulated byte. Addresses are plain Python ints.
MEM_SIZE = 512         # Total simulated bytes (0x1000 .. 0x11FF). Stack-style allocation only.
INT_SIZE = 4           # sizeof(int)   - fixed, little-endian, two's complement, signed.
PTR_SIZE = 4           # sizeof(char*) etc. - a pointer is a 4-byte unsigned address.
CHAR_SIZE = 1          # sizeof(char)  - signed 8-bit.

MAX_STEPS = 2000            # A single request may replay at most this many steps.
MAX_NODE_VISITS = 300_000   # Safety budget: stops infinite loops that never reach a step boundary.
MAX_SOURCE_CHARS = 20_000   # Reject absurdly large programs.


class MemscopeError(Exception):
    """Any error the *user's program* causes (syntax error, segfault, undeclared name...).

    These are reported to the UI as readable messages; they are not server bugs."""


# ============================================
# 3. PROGRAM STATE
# ============================================
@dataclass(frozen=True)
class CType:
    """A C type in the supported subset: int / char / void, optionally ONE level of pointer.

    `ptr=True` means "pointer to base". Pointer-to-pointer is rejected by the parser."""
    base: str
    ptr: bool = False

    def size(self) -> int:
        """Bytes this type occupies. Pointers are PTR_SIZE; int is INT_SIZE; char/void are 1."""
        if self.ptr:
            return PTR_SIZE
        return INT_SIZE if self.base == "int" else CHAR_SIZE

    def pointee(self) -> "CType":
        """The type you get by dereferencing a pointer of this type (`char*` -> `char`)."""
        return CType(self.base, False)

    def to_json(self) -> dict:
        return {"b": self.base, "ptr": self.ptr}


INT = CType("int")


@dataclass(eq=False)  # eq=False -> compared by identity, so two variables named `j` are never "equal"
class Variable:
    """One named variable = one real allocation in simulated memory.

    The variable does NOT store its value. The value lives only in memory bytes at `addr`;
    read it with Memory.read_value(). This keeps ProgramState the single source of truth.
    Arrays are one Variable with count > 1 and `is_array=True` (elements are contiguous).
    """
    name: str
    ctype: CType          # element type for arrays
    addr: int
    count: int            # number of elements (1 for scalars)
    is_array: bool
    frame_id: int
    dead: bool = False    # True once its scope ended (only kept visible for main's own variables)
    # --- reactive ("Desmos-style") bookkeeping, only set for pure declarations like `int c = a + b;` ---
    ast: Any = None                                   # initializer expression to re-evaluate
    expr: Optional[str] = None                        # initializer as text, shown in the UI
    dep_vars: list = field(default_factory=list)      # Variable objects the expression reads
    env: Optional[list] = None                        # scope chain to resolve names in when re-evaluating

    def to_json(self) -> dict:
        # Short keys on purpose: they match what the frontend renderers already read.
        return {"n": self.name, "t": self.ctype.to_json(), "a": self.addr, "cnt": self.count,
                "arr": self.is_array, "frame": self.frame_id, "dead": self.dead,
                "expr": self.expr, "deps": [v.name for v in self.dep_vars]}


@dataclass
class Frame:
    """A stack frame = one function activation.

    `scopes` is a stack of dicts name->Variable. Index 0 holds parameters and the function's own
    top-level locals; every `{ }` block and every `for` header pushes one more dict."""
    id: int
    fn: str
    scopes: list = field(default_factory=lambda: [{}])


@dataclass
class Val:
    """Result of evaluating an expression: the number, its C type, and a label for the UI
    (e.g. the variable name 'a', the text '3', or 'arr[2]'). Labels never affect execution."""
    v: int
    ctype: CType
    label: str = ""


@dataclass
class LValue:
    """'Where does this live?' - an address plus type. Produced for assignment targets, `&x`,
    `arr[i]` and `*p`. `var` is set only when the target is a plain named scalar."""
    addr: int
    ctype: CType
    label: str
    var: Optional[Variable] = None
    event: Optional[dict] = None   # POINTER_DEREFERENCE event, so a later read can attach its value


@dataclass
class ReturnSignal:
    """Returned by _exec() up the Python call chain when a `return` statement ran."""
    value: int


@dataclass
class BreakSignal:
    """Control-flow signal consumed by the nearest loop or switch statement."""


@dataclass
class ContinueSignal:
    """Control-flow signal consumed by the nearest loop (and propagated through switch)."""


class ProgramState:
    """THE single source of truth for a running program.

    Everything the UI draws comes from this object (plus the events of the latest step).
    Only the Interpreter (section 8) mutates it.
    """

    def __init__(self) -> None:
        self.memory = Memory()                  # byte-addressable simulated memory (section 4)
        self.variables: list[Variable] = []     # every variable currently alive (+ main's dimmed dead ones)
        self.frames: list[Frame] = []           # call stack; frames[-1] is the active function
        self.call_stack: list[str] = []         # function names, parallel to `frames` (for display/debug)
        self.functions: dict[str, Any] = {}     # name -> function AST node
        self.line = 0                           # 1-based source line of the statement being executed
        self.output = ""                        # everything printf has written so far (section 10)
        self.events: list[dict] = []            # events produced by the CURRENT step (section 9)
        self.done = False                       # program finished or crashed
        self.next_frame_id = 0
        self.node_visits = 0                    # budget counter, see MAX_NODE_VISITS
        # Values supplied by the browser for scanf during deterministic replay. The
        # interpreter consumes them in source order; no terminal stdin is involved.
        self.input_values: list[str] = []
        self.input_cursor = 0
        self.string_literals: dict[str, int] = {}
        self.execution_state = "RUNNING"
        self.waiting_for_input: Optional[dict] = None
        # --- transient flags used while evaluating one expression ---
        self.quiet = 0                          # >0: evaluate silently (no events); used for reactive re-evaluation
        self.index_depth = 0                    # >0: we are inside `[...]` -> arithmetic is INDEX arithmetic
        self.cond_depth = 0                     # >0: we are inside an if/for/while condition
        self.dep_collector: Optional[set] = None  # while a declaration initializer runs: collects variables it reads
        self.pure_env: Optional[list] = None    # scope chain override used during reactive re-evaluation


# ============================================
# 4. MEMORY MODEL
# ============================================
class Memory:
    """The simulated machine memory.

    ASSUMPTIONS (a teaching model - NOT identical to real C):
      * Byte-addressable: `data` is a bytearray; address A lives at data[A - MEM_BASE].
      * Little-endian: an int's least significant byte has the lowest address (130 -> 82 00 00 00).
      * int = 4 bytes signed two's complement; char = 1 byte signed; pointer = 4 bytes unsigned.
      * Addresses are small integers starting at MEM_BASE (0x1000), not real process addresses.
      * Allocation is a simple stack: `sp` only moves up when a variable is created and moves back
        down when its scope ends (see free()). There is no heap and no malloc.
      * Every allocation is rounded up to a multiple of 4 bytes, so variables start 4-byte aligned.
      * Reading uninitialised memory returns whatever bytes are there (zero at start, or stale values
        from a variable that used the same slot earlier) - just like a real stack.
    """

    def __init__(self) -> None:
        self.data = bytearray(MEM_SIZE)
        self.sp = MEM_BASE                   # next free address
        self.allocations: list[tuple[int, int]] = []   # (address, size) of live allocations

    # ---- raw access ------------------------------------------------------------------
    def _check(self, addr: int, n: int) -> None:
        """Raise a 'segmentation fault' if [addr, addr+n) is outside simulated memory."""
        if addr < MEM_BASE or addr + n > MEM_BASE + MEM_SIZE:
            raise MemscopeError(f"Segmentation fault: {hex(addr)} is not memory this program owns")

    def read_byte(self, addr: int) -> int:
        """Return the unsigned byte (0-255) at addr."""
        self._check(addr, 1)
        return self.data[addr - MEM_BASE]

    def write_byte(self, addr: int, value: int) -> None:
        """Store `value` (wrapped to 0-255) at addr."""
        self._check(addr, 1)
        self.data[addr - MEM_BASE] = value % 256

    def read_unsigned(self, addr: int, n: int) -> int:
        """Little-endian unsigned integer made of n bytes starting at addr."""
        self._check(addr, n)
        return int.from_bytes(self.data[addr - MEM_BASE: addr - MEM_BASE + n], "little")

    def write_unsigned(self, addr: int, n: int, value: int) -> None:
        """Write value into n bytes, little-endian. Out-of-range values wrap (like C integer overflow)."""
        self._check(addr, n)
        self.data[addr - MEM_BASE: addr - MEM_BASE + n] = (value % (1 << (8 * n))).to_bytes(n, "little")

    # ---- typed access ----------------------------------------------------------------
    def read_int(self, addr: int) -> int:
        """Signed 4-byte int."""
        u = self.read_unsigned(addr, INT_SIZE)
        return u - (1 << 32) if u >= 1 << 31 else u

    def write_int(self, addr: int, value: int) -> None:
        self.write_unsigned(addr, INT_SIZE, value)

    def read_pointer(self, addr: int) -> int:
        """A pointer is just an unsigned 4-byte address."""
        return self.read_unsigned(addr, PTR_SIZE)

    def write_pointer(self, addr: int, target: int) -> None:
        self.write_unsigned(addr, PTR_SIZE, target)

    def read_value(self, ctype: CType, addr: int) -> int:
        """Read a value of `ctype` stored at addr (this is THE read path used by the interpreter)."""
        if ctype.ptr:
            return self.read_pointer(addr)
        if ctype.base == "char":
            b = self.read_byte(addr)
            return b - 256 if b > 127 else b       # signed char
        return self.read_int(addr)

    def write_value(self, ctype: CType, addr: int, value: int) -> None:
        """Write a value of `ctype` at addr (THE write path used by the interpreter)."""
        if ctype.ptr:
            self.write_pointer(addr, value)
        elif ctype.base == "char":
            self.write_byte(addr, value)
        else:
            self.write_int(addr, value)

    # ---- allocation ------------------------------------------------------------------
    def allocate(self, size: int) -> int:
        """Reserve `size` bytes (rounded up to a multiple of 4) at the stack pointer. Returns the address."""
        addr = self.sp
        self._check(addr, size)
        self.sp += max(4, -(-size // 4) * 4)
        self.allocations.append((addr, size))
        return addr

    def free(self, addr: int) -> None:
        """Release the allocation at `addr` AND everything allocated after it (stack discipline).

        Bytes are NOT erased - the next allocation reusing this slot will see the old contents until
        it writes its own value, exactly like a real stack."""
        self.sp = addr
        self.allocations = [(a, s) for (a, s) in self.allocations if a < addr]


# ============================================
# 5. VARIABLES AND SCOPES
# ============================================
# LIFETIME RULES (this is what prevents "a new `j` on every loop iteration"):
#   * A declaration (`int j = 0;`) runs create_variable() ONCE per scope entry: it allocates memory
#     and registers the name in the innermost scope dict of the current frame.
#   * Later statements (`j = 1;`, `j++`) only LOOK UP the existing variable and write its memory.
#   * A `for (int j = ...)` header runs its init once per execution of the loop, not once per iteration.
#   * When a scope ends, pop_scope() removes its variables and rewinds the stack pointer. Re-entering
#     the same code later (e.g. an inner loop on every outer iteration) therefore reuses the SAME
#     addresses, as a real stack frame does.

def lookup_variable(state: ProgramState, name: str) -> Variable:
    """Resolve `name` by searching scopes from innermost to outermost in the active frame, then the
    global scope (frames[0].scopes[0]). Uses `state.pure_env` instead when set (reactive re-evaluation).
    Raises MemscopeError if undeclared. Does not change state."""
    scopes = state.pure_env if state.pure_env is not None else state.frames[-1].scopes
    for scope in reversed(scopes):
        if name in scope:
            return scope[name]
    global_scope = state.frames[0].scopes[0]
    if name in global_scope:
        return global_scope[name]
    raise MemscopeError(f"{name} is not declared")


def display_value(state: ProgramState, var: Variable) -> str:
    """Human-readable current value, read from memory: '130', '0x1000' (pointer) or '{10, 20}' (array)."""
    mem = state.memory
    if var.is_array:
        step = var.ctype.size()
        return "{" + ", ".join(str(mem.read_value(var.ctype, var.addr + k * step)) for k in range(var.count)) + "}"
    value = mem.read_value(var.ctype, var.addr)
    return hex(value) if var.ctype.ptr else str(value)


def create_variable(state: ProgramState, frame: Frame, name: str, ctype: CType, count: int,
                    is_array: bool, init_values: Optional[list]) -> Variable:
    """Declare a variable in the INNERMOST scope of `frame`.

    Steps: refuse duplicates in the same scope -> allocate bytes -> register the name -> write the
    initial values (if any) -> emit MEMORY_ALLOCATE / MEMORY_WRITE / VARIABLE_CREATE / POINTER_CREATE.
    Changes: memory, state.variables, the scope dict, state.events. Returns the new Variable."""
    scope = frame.scopes[-1]
    if name in scope:
        raise MemscopeError(f"{name} is already declared")
    size = ctype.size() * count
    addr = state.memory.allocate(size)
    var = Variable(name, ctype, addr, count, is_array, frame.id)
    emit(state, "MEMORY_ALLOCATE", addr=addr, size=size, name=name)
    scope[name] = var
    state.variables.append(var)
    for k, value in enumerate(init_values or []):
        a = addr + k * ctype.size()
        state.memory.write_value(ctype, a, value)
        emit(state, "MEMORY_WRITE", init=True, addr=a, size=ctype.size(),
             value=state.memory.read_value(ctype, a),          # re-read: shows the wrapped/stored value
             target=name + (f"[{k}]" if is_array else ""), **{"from": ""})
    emit(state, "VARIABLE_CREATE", name=name, address=addr, size=size,
         shown=display_value(state, var) if init_values else None)
    if ctype.ptr and init_values:
        emit(state, "POINTER_CREATE", name=name, target=state.memory.read_value(ctype, addr))
    return var


def pop_scope(state: ProgramState, frame: Frame, keep_visible: bool = False) -> None:
    """End the innermost scope of `frame`.

    All its variables are marked dead. Unless keep_visible, they are also removed from
    state.variables and the stack pointer is rewound to the lowest of their addresses.
    keep_visible=True is used only for main's own frame so the final memory stays visible."""
    scope = frame.scopes.pop()
    for var in scope.values():
        var.dead = True
    if keep_visible or not scope:
        return
    doomed = list(scope.values())
    state.variables[:] = [v for v in state.variables if v not in doomed]   # identity comparison
    state.memory.free(min(v.addr for v in doomed))


# ============================================
# 6. LEXER / TOKENIZER
# ============================================
@dataclass
class Token:
    text: str
    kind: str   # 'num' | 'id' | 'str' | 'chr' | 'op'
    line: int


# One alternation, tried left to right. Order matters: comments must come before the '/' operator and
# two-character operators before one-character ones.
_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<preproc>\#[^\n]*)
  | (?P<num>\d+)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<chr>'(?:[^'\\]|\\.)')
  | (?P<op>\+\+|--|<=|>=|==|!=|&&|\|\||[-+*/%]=|[-+*/%&<>=!(){}\[\];,:])
""", re.X | re.S)


def tokenize(source: str) -> list[Token]:
    """Turn source text into tokens. Whitespace, comments and `#include` lines are dropped.
    Tracks 1-based line numbers (newlines inside skipped text are counted). Raises MemscopeError
    on a character that cannot start any token."""
    tokens: list[Token] = []
    pos, line = 0, 1
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            raise MemscopeError(f"Unexpected '{source[pos]}' on line {line}")
        kind, text = m.lastgroup, m.group()
        if kind not in ("ws", "comment", "preproc"):
            tokens.append(Token(text, kind, line))
        line += text.count("\n")
        pos = m.end()
    return tokens


# ============================================
# 7. PARSER / AST
# ============================================
def node(kind: str, line: int, **fields) -> SimpleNamespace:
    """Create an AST node. Every node has `.k` (kind) and `.line`; other fields depend on the kind:

      Expressions                                  Statements
      Num(v)  Str(v)  Var(name)                    Block(body)            If(cond, then, other)
      Un(op, operand)   Cast(ctype, operand)       While(cond, body)      DoWhile(body, cond)
      Bin(op, left, right)                         For(init, cond, update, body)
                                                    Switch(expr, cases)    Break / Continue
                                                    Return(value)          Expr(expr)
      Assign(target, value)                        Decl(ctype, name, size, init, no_react)
      IncDec(op, operand, post)                    InitList(items)  (only inside Decl.init)
      Index(base, index)   Call(name, args)

    Expression nodes that are the ROOT of a condition or initializer also get `.text` (source text),
    used only for UI messages such as 'i < n - 1'."""
    return SimpleNamespace(k=kind, line=line, **fields)


class Parser:
    """Recursive-descent parser: tokens -> AST. It never executes anything and never touches state.

    Each precedence level of C expressions is one method, lowest to highest:
      assignment -> || -> && -> == != -> < > <= >= -> + - -> * / % -> unary -> postfix -> primary
    Compound assignment `a += b` is desugared to `a = a + b`.
    """

    TYPES = ("int", "char", "void")
    BINARY_LEVELS = [("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%")]

    def __init__(self, tokens: list[Token]) -> None:
        self.t = tokens
        self.i = 0

    # ---- token helpers ----
    def peek(self) -> Token:
        if self.i >= len(self.t):
            raise MemscopeError("Unexpected end of code")
        return self.t[self.i]

    def next(self) -> Token:
        tok = self.peek()
        self.i += 1
        return tok

    def is_(self, text: str) -> bool:
        return self.i < len(self.t) and self.t[self.i].text == text

    def accept(self, text: str) -> bool:
        if self.is_(text):
            self.i += 1
            return True
        return False

    def expect(self, text: str) -> None:
        if not self.accept(text):
            tok = self.t[self.i] if self.i < len(self.t) else self.t[-1]
            found = f"'{tok.text}'" if self.i < len(self.t) else "end of code"
            raise MemscopeError(f"Expected '{text}' but found {found} on line {tok.line}")

    def is_type(self) -> bool:
        return self.i < len(self.t) and self.t[self.i].text in self.TYPES

    def parse_type(self) -> CType:
        base = self.next().text
        ptr = False
        while self.accept("*"):
            if ptr:
                raise MemscopeError("Only single-level pointers are supported")
            ptr = True
        return CType(base, ptr)

    # ---- program ----
    def parse_program(self) -> SimpleNamespace:
        """Returns Program(top=[statements outside any function], funcs={name: function node}).
        Bare statements are allowed at top level so tiny snippets run without a main()."""
        top, funcs = [], {}
        while self.i < len(self.t):
            is_function = (self.is_type() and self.i + 2 < len(self.t)
                           and self.t[self.i + 1].kind == "id" and self.t[self.i + 2].text == "(")
            if not is_function:
                top.append(self.statement())
                continue
            line = self.t[self.i].line
            ret = self.parse_type()
            name = self.next().text
            self.expect("(")
            params = []
            if not self.is_(")"):
                if self.is_("void"):
                    self.next()
                else:
                    while True:
                        ptype = self.parse_type()
                        params.append((ptype, self.next().text))
                        if not self.accept(","):
                            break
            self.expect(")")
            funcs[name] = node("Func", line, name=name, ret=ret, params=params, body=self.statement())
        return SimpleNamespace(top=top, funcs=funcs)

    # ---- expressions ----
    def expression(self) -> SimpleNamespace:
        """Parse an assignment-level expression and remember its source text on the node."""
        start = self.i
        e = self.assignment()
        e.text = " ".join(t.text for t in self.t[start:self.i])
        return e

    def assignment(self) -> SimpleNamespace:
        left = self.binary(0)
        if self.i < len(self.t) and re.fullmatch(r"[-+*/%]?=", self.t[self.i].text):
            op = self.next()
            right = self.assignment()
            if op.text != "=":   # a += b  ==>  a = a + b
                right = node("Bin", op.line, op=op.text[0], left=left, right=right)
            return node("Assign", op.line, target=left, value=right)
        return left

    def binary(self, level: int) -> SimpleNamespace:
        if level >= len(self.BINARY_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.i < len(self.t) and self.t[self.i].text in self.BINARY_LEVELS[level]:
            op = self.next()
            left = node("Bin", op.line, op=op.text, left=left, right=self.binary(level + 1))
        return left

    def unary(self) -> SimpleNamespace:
        tok = self.peek()
        if tok.text in ("-", "!", "*", "&"):
            self.i += 1
            return node("Un", tok.line, op=tok.text, operand=self.unary())
        if tok.text in ("++", "--"):
            self.i += 1
            return node("IncDec", tok.line, op=tok.text, operand=self.unary(), post=False)
        if tok.text == "(" and self.i + 1 < len(self.t) and self.t[self.i + 1].text in self.TYPES:
            self.i += 1                        # a cast such as (char *)&a
            ctype = self.parse_type()
            self.expect(")")
            return node("Cast", tok.line, ctype=ctype, operand=self.unary())
        return self.postfix()

    def postfix(self) -> SimpleNamespace:
        e = self.primary()
        while self.i < len(self.t):
            tok = self.t[self.i]
            if tok.text == "[":
                self.i += 1
                idx = self.assignment()
                self.expect("]")
                e = node("Index", tok.line, base=e, index=idx)
            elif tok.text in ("++", "--"):
                self.i += 1
                e = node("IncDec", tok.line, op=tok.text, operand=e, post=True)
            else:
                break
        return e

    def primary(self) -> SimpleNamespace:
        tok = self.next()
        if tok.kind == "num":
            return node("Num", tok.line, v=int(tok.text))
        if tok.kind == "chr":
            body = tok.text[1:-1]
            escapes = {"n": 10, "t": 9, "0": 0, "\\": 92, "'": 39}
            return node("Num", tok.line, v=escapes.get(body[1], 0) if body[0] == "\\" else ord(body))
        if tok.kind == "str":
            return node("Str", tok.line, v=tok.text[1:-1])
        if tok.text == "(":
            e = self.assignment()
            self.expect(")")
            return e
        if tok.kind == "id":
            if self.accept("("):
                args = []
                if not self.is_(")"):
                    while True:
                        args.append(self.assignment())
                        if not self.accept(","):
                            break
                self.expect(")")
                return node("Call", tok.line, name=tok.text, args=args)
            return node("Var", tok.line, name=tok.text)
        raise MemscopeError(f"Unexpected '{tok.text}' on line {tok.line}")

    # ---- statements ----
    def declaration(self, no_react: bool = False) -> SimpleNamespace:
        """Parse scalar/array declarations, including comma-separated declarators.
        `no_react=True` is used for `for (int i = ...)` so the loop counter is never reactive."""
        line = self.t[self.i].line
        ctype = self.parse_type()
        decls = []
        while True:
            name = self.next()
            if name.kind != "id":
                raise MemscopeError(f"Expected a name on line {name.line}")
            size = None
            if self.accept("["):
                size = 0 if self.is_("]") else int(self.next().text)
                self.expect("]")
            init = None
            if self.accept("="):
                if self.is_("{"):
                    self.i += 1
                    items = []
                    while not self.is_("}"):
                        items.append(self.assignment())
                        if not self.accept(","):
                            break
                    self.expect("}")
                    init = node("InitList", line, items=items)
                else:
                    init = self.expression()
            decls.append(node("Decl", line, ctype=ctype, name=name.text, size=size,
                              init=init, no_react=no_react))
            if not self.accept(","):
                break
        self.expect(";")
        return decls[0] if len(decls) == 1 else node("DeclGroup", line, decls=decls)

    def expression_statement(self) -> SimpleNamespace:
        line = self.peek().line
        e = self.assignment()
        self.expect(";")
        return node("Expr", line, expr=e)

    def statement(self) -> SimpleNamespace:
        tok = self.peek()
        if tok.text == "{":
            self.i += 1
            body = []
            while not self.is_("}"):
                if self.i >= len(self.t):
                    raise MemscopeError("Missing '}'")
                body.append(self.statement())
            self.i += 1
            return node("Block", tok.line, body=body)
        if tok.text == "if":
            self.i += 1
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            then = self.statement()
            other = self.statement() if self.accept("else") else None
            return node("If", tok.line, cond=cond, then=then, other=other)
        if tok.text == "switch":
            self.i += 1
            self.expect("(")
            expr = self.expression()
            self.expect(")")
            self.expect("{")
            cases = []
            current = None
            # Group statements under labels. Execution can start at a matching label
            # and continue through later groups to model C's fall-through behavior.
            while not self.is_("}"):
                if self.i >= len(self.t):
                    raise MemscopeError("Missing '}'")
                if self.accept("case"):
                    label = self.t[self.i - 1]
                    value = self.assignment()  # stops before ':'; colon is a label delimiter
                    self.expect(":")
                    current = node("Case", label.line, value=value, body=[])
                    cases.append(current)
                elif self.accept("default"):
                    label = self.t[self.i - 1]
                    self.expect(":")
                    current = node("Case", label.line, value=None, body=[])
                    cases.append(current)
                else:
                    if current is None:
                        raise MemscopeError(f"Expected 'case' or 'default' on line {self.peek().line}")
                    current.body.append(self.statement())
            self.i += 1
            return node("Switch", tok.line, expr=expr, cases=cases)
        if tok.text == "break":
            self.i += 1
            self.expect(";")
            return node("Break", tok.line)
        if tok.text == "continue":
            self.i += 1
            self.expect(";")
            return node("Continue", tok.line)
        if tok.text == "while":
            self.i += 1
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            return node("While", tok.line, cond=cond, body=self.statement())
        if tok.text == "do":
            self.i += 1
            body = self.statement()
            if not self.accept("while"):
                line = self.peek().line if self.i < len(self.t) else tok.line
                raise MemscopeError(f"Expected 'while' after do body on line {line}")
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            self.expect(";")
            return node("DoWhile", tok.line, body=body, cond=cond)
        if tok.text == "for":
            self.i += 1
            self.expect("(")
            init = None
            if not self.accept(";"):
                init = self.declaration(no_react=True) if self.is_type() else self.expression_statement()
            cond = None if self.is_(";") else self.expression()
            self.expect(";")
            update = None if self.is_(")") else self.assignment()
            self.expect(")")
            return node("For", tok.line, init=init, cond=cond, update=update, body=self.statement())
        if tok.text == "return":
            self.i += 1
            value = None if self.is_(";") else self.expression()
            self.expect(";")
            return node("Return", tok.line, value=value)
        if self.is_type():
            return self.declaration()
        if self.accept(";"):
            return node("Block", tok.line, body=[])
        return self.expression_statement()


# ============================================
# 8. EXECUTION ENGINE
# ============================================
# The interpreter is a set of Python GENERATORS. Every `yield` is a step boundary: the point where the
# UI may pause and show what just happened. step() resumes the generator until its next `yield`.
#
# Where do steps end?
#   * a declaration            -> 1 step (after memory is written)
#   * `x = e;` / `a[i] = e;`   -> 2 steps if evaluating needed any memory reads (compute, then write), else 1
#   * `i++`, `n--`             -> 1 step
#   * if                        -> 1 step after the condition is evaluated
#   * for/while                 -> a step at loop exit; iterations are separated by their body/update steps
#   * a call to printf         -> 1 step; a user function call -> 1 step on entry and 1 on return
#
# _eval  = "what is its VALUE?"  (reads memory)       _lvalue = "WHERE does it live?" (an address)

_ARITHMETIC = "+-*/%"


def c_div(a: int, b: int) -> int:
    """C integer division truncates toward zero (Python's // floors, so we cannot use it directly)."""
    if b == 0:
        raise MemscopeError("Division by zero")
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def c_mod(a: int, b: int) -> int:
    """C remainder: a == (a / b) * b + (a % b), so the result takes the sign of the dividend."""
    return a - b * c_div(a, b)


OPERATORS = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": c_div, "%": c_mod,
    "<": lambda a, b: int(a < b), ">": lambda a, b: int(a > b),
    "<=": lambda a, b: int(a <= b), ">=": lambda a, b: int(a >= b),
    "==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b),
}


def is_pure(n) -> bool:
    """True if evaluating `n` cannot change program state (no assignment, ++/--, or call).
    Only pure initializers may become reactive formulas."""
    if n is None:
        return True
    k = n.k
    if k in ("Num", "Var"):
        return True
    if k == "Bin":
        return is_pure(n.left) and is_pure(n.right)
    if k in ("Un", "Cast"):
        return is_pure(n.operand)
    if k == "Index":
        return is_pure(n.base) and is_pure(n.index)
    return False


def source_text(n) -> str:
    """Rebuild readable text for a (small) expression, e.g. `i + 1`. UI labels only."""
    k = n.k
    if k == "Num":
        return str(n.v)
    if k == "Var":
        return n.name
    if k == "Bin":
        return f"{source_text(n.left)} {n.op} {source_text(n.right)}"
    if k == "Un":
        return n.op + source_text(n.operand)
    if k == "Index":
        return f"{source_text(n.base)}[{source_text(n.index)}]"
    return "\u2026"


class Interpreter:
    """Runs one program. Owns the ProgramState and the step generator.

    Typical use (see run_program):
        interp = Interpreter(program)
        events = interp.step()      # one action; returns that step's events, or None when finished
        interp.state                # inspect memory / variables / output after the step
    """

    def __init__(self, program: SimpleNamespace, input_values: Optional[list[str]] = None) -> None:
        self.state = ProgramState()
        self.state.input_values = list(input_values or [])
        self._steps = self._run_program(program)

    # ---- public API --------------------------------------------------------------------
    def step(self) -> Optional[list[dict]]:
        """Advance the program by ONE step.

        Changes: ProgramState (memory, variables, frames, output) and `state.events`.
        Returns: the list of events produced by this step, or None if the program had already finished
        (or finished without doing anything more). A runtime error becomes an ERROR event and finishes
        the program."""
        s = self.state
        if s.done:
            return None
        s.events = []
        try:
            next(self._steps)
        except StopIteration:
            s.done = True
        except MemscopeError as err:
            s.done = True
            s.events.append({"type": "ERROR", "message": str(err), "line": s.line})
        return s.events if s.events else None

    # ---- bookkeeping -------------------------------------------------------------------
    def _visit(self) -> None:
        """Count node visits so a program that loops forever without reaching a step boundary
        (e.g. `while (1) {}`) is stopped instead of hanging the server."""
        self.state.node_visits += 1
        if self.state.node_visits > MAX_NODE_VISITS:
            raise MemscopeError("Program ran too long without finishing a step (infinite loop?)")

    def _run_pure(self, expr) -> int:
        """Evaluate a pure expression to completion with NO events (state.quiet). Used to recompute
        reactive variables. Never yields to the caller."""
        self.state.quiet += 1
        try:
            gen = self._eval(expr)
            try:
                while True:
                    next(gen)
            except StopIteration as done:
                return done.value.v
        finally:
            self.state.quiet -= 1

    # ---- program / functions -----------------------------------------------------------
    def _run_program(self, program) -> Generator:
        """Top-level driver. Runs top-level statements, then main() if it exists.
        If main() exists the call is 'silent' (no step on entry) so the first step is main's first
        statement. Without main(), top-level code runs in an implicit frame named main."""
        s = self.state
        s.functions = program.funcs
        if program.top or "main" not in program.funcs:
            s.next_frame_id += 1
            s.frames.append(Frame(s.next_frame_id, "globals" if "main" in program.funcs else "main"))
        for stmt in program.top:
            result = yield from self._exec(stmt)
            if isinstance(result, (BreakSignal, ContinueSignal)):
                raise MemscopeError("break/continue is only valid inside a loop or switch")
        if "main" in program.funcs:
            yield from self._call("main", [], silent=True)

    def _call(self, name: str, args: list[Val], silent: bool = False) -> Generator:
        """Call a user-defined function: push a Frame, create parameters (by value), run the body in the
        frame's first scope, pop the frame. Emits FUNCTION_CALL / FUNCTION_RETURN. Returns Val.
        main's variables stay visible (dimmed) after it returns so the final memory can still be inspected."""
        s = self.state
        fn = s.functions.get(name)
        if fn is None:
            raise MemscopeError(f"Unknown function {name}()")
        if len(args) != len(fn.params):
            raise MemscopeError(f"{name}() expects {len(fn.params)} argument(s)")
        s.next_frame_id += 1
        frame = Frame(s.next_frame_id, name)
        s.frames.append(frame)
        s.call_stack.append(name)
        emit(s, "FUNCTION_CALL", name=name, args=[a.v for a in args], implicit=silent)
        for (ptype, pname), arg in zip(fn.params, args):
            create_variable(s, frame, pname, ptype, 1, False, [arg.v])
        if not silent:
            yield
        result = None
        s.line = fn.body.line
        for stmt in fn.body.body:           # run body statements directly in the parameter scope
            result = yield from self._exec(stmt)
            if result:
                break
        value = result.value if result else 0
        if isinstance(result, (BreakSignal, ContinueSignal)):
            raise MemscopeError("break/continue is only valid inside a loop or switch")
        pop_scope(s, frame, keep_visible=(name == "main"))
        s.frames.pop()
        s.call_stack.pop()
        emit(s, "FUNCTION_RETURN", name=name, value=value)
        yield
        return Val(value, fn.ret, f"{name}()")

    # ---- statements --------------------------------------------------------------------
    def _condition(self, expr, show: bool = False) -> Generator:
        """Evaluate a condition. Arithmetic inside it is tagged CONDITION_ARITHMETIC (not a value
        computation). Emits CONDITION_EVALUATED. Returns bool."""
        s = self.state
        s.cond_depth += 1
        try:
            r = yield from self._eval(expr)
        finally:
            s.cond_depth -= 1
        emit(s, "CONDITION_EVALUATED", expr=expr.text, value=bool(r.v), show=show)
        return bool(r.v)

    def _exec(self, n) -> Generator:
        """Execute ONE statement node. Returns None, or a ReturnSignal if `return` executed
        (callers propagate it upward until _call consumes it)."""
        s = self.state
        self._visit()
        s.line = n.line
        frame = s.frames[-1]
        k = n.k

        if k == "Block":
            frame.scopes.append({})
            try:
                for stmt in n.body:
                    r = yield from self._exec(stmt)
                    if r:
                        return r
            finally:
                pop_scope(s, frame)         # always, even on return: ends the block's variables
            return None

        if k == "DeclGroup":
            for decl in n.decls:
                result = yield from self._exec(decl)
                if result is not None:
                    return result
            return None

        if k == "Decl":
            # 1. Evaluate the initializer FIRST (reads happen before the new variable exists).
            #    dep_collector records which variables it reads -> reactive dependencies.
            s.dep_collector = set()
            values = None
            if n.init is not None:
                if n.init.k == "Str" and n.ctype.base == "char" and n.size is not None:
                    values = [ord(ch) for ch in decode_c_escapes(n.init.v)] + [0]
                elif n.init.k == "InitList":
                    values = []
                    for item in n.init.items:
                        values.append((yield from self._eval(item)).v)
                else:
                    values = [(yield from self._eval(n.init)).v]
            deps, s.dep_collector = s.dep_collector, None
            is_array = n.size is not None
            count = (n.size or len(values or [])) if is_array else 1
            if is_array and count == 0:
                raise MemscopeError("Array needs a size")
            if values and len(values) > count:
                raise MemscopeError("Too many initializers")
            mark_destination(s, n.name)     # lets the UI draw "... -> n.name" on the arithmetic that fed it
            var = create_variable(s, frame, n.name, n.ctype, count, is_array, values)
            if (values and not n.no_react and not is_array and not n.ctype.ptr and deps and is_pure(n.init)):
                var.ast, var.expr = n.init, n.init.text
                var.dep_vars, var.env = list(deps), list(frame.scopes)
            yield
            return None

        if k == "Expr":
            yield from self._eval(n.expr)
            return None

        if k == "Return":
            r = (yield from self._eval(n.value)) if n.value is not None else Val(0, INT)
            emit(s, "RETURN", value=r.v)
            return ReturnSignal(r.v)

        if k == "Break":
            emit(s, "BREAK")
            yield
            return BreakSignal()

        if k == "Continue":
            emit(s, "CONTINUE")
            yield
            return ContinueSignal()

        if k == "If":
            taken = yield from self._condition(n.cond, show=True)
            yield
            if taken:
                return (yield from self._exec(n.then))
            if n.other is not None:
                return (yield from self._exec(n.other))
            return None

        if k == "While":
            emit(s, "LOOP_ENTER", kind="while")
            yield
            count = 0
            while True:
                if not (yield from self._condition(n.cond)):
                    emit(s, "LOOP_EXIT", text=n.cond.text)
                    yield
                    return None
                count += 1
                emit(s, "LOOP_ITERATION", n=count, text=n.cond.text)
                r = yield from self._exec(n.body)
                if isinstance(r, ReturnSignal):
                    return r
                if isinstance(r, BreakSignal):
                    emit(s, "LOOP_BREAK")
                    yield
                    return None
                if isinstance(r, ContinueSignal):
                    s.line = n.line
                    continue
                s.line = n.line

        if k == "DoWhile":
            emit(s, "LOOP_ENTER", kind="do-while")
            yield
            count = 0
            while True:
                count += 1
                emit(s, "LOOP_ITERATION", n=count, text=n.cond.text)
                r = yield from self._exec(n.body)
                if isinstance(r, ReturnSignal):
                    return r
                if isinstance(r, BreakSignal):
                    emit(s, "LOOP_BREAK")
                    yield
                    return None
                # Continue reaches the condition test, as required by do-while semantics.
                if not (yield from self._condition(n.cond)):
                    emit(s, "LOOP_EXIT", text=n.cond.text)
                    yield
                    return None

        if k == "For":
            # The for-header gets its own scope so `int i` lives exactly as long as the loop.
            # The init statement runs ONCE here; iterations only run cond -> body -> update.
            frame.scopes.append({})
            try:
                emit(s, "LOOP_ENTER", kind="for")
                yield
                if n.init is not None:
                    yield from self._exec(n.init)
                count = 0
                while True:
                    ok = (yield from self._condition(n.cond)) if n.cond is not None else True
                    if not ok:
                        emit(s, "LOOP_EXIT", text=n.cond.text if n.cond is not None else "")
                        yield
                        return None
                    count += 1
                    emit(s, "LOOP_ITERATION", n=count, text=n.cond.text if n.cond is not None else "")
                    r = yield from self._exec(n.body)
                    if isinstance(r, ReturnSignal):
                        return r
                    if isinstance(r, BreakSignal):
                        emit(s, "LOOP_BREAK")
                        yield
                        return None
                    if isinstance(r, ContinueSignal):
                        # A continue in a for body still executes the update clause.
                        if n.update is not None:
                            s.line = n.line
                            yield from self._eval(n.update)
                        continue
                    if n.update is not None:
                        s.line = n.line         # the update belongs to the `for` line
                        yield from self._eval(n.update)
            finally:
                pop_scope(s, frame)

        if k == "Switch":
            # Evaluate the selector exactly once. Case expressions are constants, so
            # evaluate them quietly without generating side effects or memory events.
            selector = yield from self._eval(n.expr)
            emit(s, "SWITCH_ENTER", expr=n.expr.text, value=selector.v)
            yield
            selected = None
            default_index = None
            for index, case in enumerate(n.cases):
                if case.value is None:
                    default_index = index
                    continue
                case_value = self._run_pure(case.value)
                emit(s, "CASE_CHECK", value=case_value, switch_value=selector.v)
                yield
                if selected is None and case_value == selector.v:
                    selected = index
                    emit(s, "CASE_MATCH", value=case_value, index=index)
                    yield
                    break
            if selected is None:
                selected = default_index
            if selected is not None:
                frame.scopes.append({})
                try:
                    # Do not stop at the next label: reaching it is C-style fall-through.
                    for index in range(selected, len(n.cases)):
                        case = n.cases[index]
                        emit(s, "CASE_EXECUTION", index=index, default=case.value is None)
                        yield
                        for stmt in case.body:
                            result = yield from self._exec(stmt)
                            if isinstance(result, ReturnSignal):
                                return result
                            if isinstance(result, BreakSignal):
                                emit(s, "SWITCH_EXIT", reason="break")
                                yield
                                return None
                            if isinstance(result, ContinueSignal):
                                # continue belongs to an enclosing loop, so preserve it.
                                return result
                finally:
                    pop_scope(s, frame)
            emit(s, "SWITCH_EXIT", reason="complete")
            yield
            return None

        raise MemscopeError(f"Unsupported statement {k}")

    # ---- reactivity --------------------------------------------------------------------
    def _propagate(self, changed: set) -> list[dict]:
        """Recompute every reactive variable whose formula depends (directly or through other reactive
        variables) on something in `changed` (a set of Variable objects). Variables are visited in
        declaration order, which is always a valid dependency order. Writes the new values to MEMORY.
        Returns [{name, old, new}] for variables whose value actually changed."""
        s = self.state
        log = []
        for var in list(s.variables):
            if var.ast is not None and any(d in changed for d in var.dep_vars):
                old = s.memory.read_value(var.ctype, var.addr)
                s.pure_env = var.env
                try:
                    new = self._run_pure(var.ast)
                finally:
                    s.pure_env = None
                s.memory.write_value(var.ctype, var.addr, new)
                stored = s.memory.read_value(var.ctype, var.addr)
                if stored != old:
                    log.append({"name": var.name, "old": old, "new": stored})
                    changed.add(var)
        return log

    def _after_write(self, var: Optional[Variable]) -> None:
        """Called after a named scalar changed by assignment/++/--: detach it from any formula it had
        (an assignment overrides the formula) and recalculate its dependents (REACTIVE_UPDATE events)."""
        if var is None:
            return
        var.ast = None
        for item in self._propagate({var}):
            emit(self.state, "REACTIVE_UPDATE", name=item["name"], old=item["old"], value=item["new"])

    def _string_value(self, pointer: Val) -> str:
        """Read a null-terminated char pointer through ProgramState memory."""
        if not pointer.ctype.ptr or pointer.ctype.base != "char":
            raise MemscopeError("Expected a char pointer or character array")
        chars = []
        for offset in range(MEM_SIZE):
            addr = pointer.v + offset
            byte = self.state.memory.read_byte(addr)
            emit(self.state, "MEMORY_READ", addr=addr, size=1, value=byte,
                 source=f"{pointer.label}[{offset}]")
            if byte == 0:
                break
            chars.append(chr(byte))
        return "".join(chars)

    # ---- expressions -------------------------------------------------------------------
    def _lvalue(self, n, rw: str) -> Generator:
        """Compute WHERE `n` lives. rw is 'r' (about to be read) or 'w' (about to be written); it is
        only recorded on events so the UI can colour source vs destination.

        Supports: variable, array/pointer indexing `b[i]`, and dereference `*p`.
        Events: ARRAY_INDEX for `b[i]`, POINTER_DEREFERENCE for `*p`. Returns LValue."""
        s = self.state
        self._visit()
        if n.k == "Var":
            var = lookup_variable(s, n.name)
            if var.is_array and rw == "w":
                raise MemscopeError(f"Can't assign to array {var.name}")
            # An array name used as an address decays to a pointer to its first element.
            ctype = CType(var.ctype.base, True) if var.is_array else var.ctype
            return LValue(var.addr, ctype, var.name, None if var.is_array else var)
        if n.k == "Index":
            base = yield from self._eval(n.base)
            s.index_depth += 1             # arithmetic in the index computes a POSITION, not a stored value
            try:
                idx = yield from self._eval(n.index)
            finally:
                s.index_depth -= 1
            if not base.ctype.ptr:
                raise MemscopeError("Only arrays and pointers can be indexed")
            elem = base.ctype.pointee()
            addr = base.v + idx.v * elem.size()
            emit(s, "ARRAY_INDEX", array=base.label, index=idx.v, addr=addr, rw=rw, expr=source_text(n.index))
            return LValue(addr, elem, f"{base.label}[{idx.v}]")
        if n.k == "Un" and n.op == "*":
            ptr = yield from self._eval(n.operand)
            if not ptr.ctype.ptr:
                raise MemscopeError("Cannot dereference a non-pointer")
            elem = ptr.ctype.pointee()
            ev = emit(s, "POINTER_DEREFERENCE", ptr=ptr.label, addr=ptr.v, size=elem.size(), rw=rw)
            return LValue(ptr.v, elem, "*" + ptr.label, None, ev)
        raise MemscopeError("Cannot assign to this expression")

    def _eval(self, n) -> Generator:
        """Evaluate expression `n` and return a Val.

        Roughly: resolve variables -> read memory when a value is needed -> apply operators -> return.
        Assignments and ++/-- WRITE memory here and yield (they are steps). Everything that touches
        memory goes through state.memory, so ProgramState stays authoritative."""
        s = self.state
        self._visit()
        k = n.k

        # a[i] and *p used as VALUES: find the address, then read it.
        if k == "Index" or (k == "Un" and n.op == "*"):
            lv = yield from self._lvalue(n, "r")
            value = s.memory.read_value(lv.ctype, lv.addr)
            if lv.event is not None:
                lv.event["value"] = value
            emit(s, "MEMORY_READ", addr=lv.addr, size=lv.ctype.size(), value=value, source=lv.label)
            return Val(value, lv.ctype, lv.label)

        if k == "Num":
            return Val(n.v, INT, str(n.v))

        if k == "Str":
            # String literals are immutable char bytes in simulated memory, so library
            # calls can use the same pointer/array paths as user-created char arrays.
            if n.v not in s.string_literals:
                raw = decode_c_escapes(n.v).encode("latin1") + b"\0"
                addr = s.memory.allocate(len(raw))
                s.string_literals[n.v] = addr
                emit(s, "MEMORY_ALLOCATE", addr=addr, size=len(raw), name="string literal")
                for offset, byte in enumerate(raw):
                    s.memory.write_byte(addr + offset, byte)
                    emit(s, "MEMORY_WRITE", addr=addr + offset, size=1, value=byte,
                         target="string literal", **{"from": "literal"})
            addr = s.string_literals[n.v]
            return Val(addr, CType("char", True), '"' + n.v + '"')

        if k == "Var":
            var = lookup_variable(s, n.name)
            if s.dep_collector is not None:
                s.dep_collector.add(var)
            if var.is_array:                 # array name -> address of element 0 (no read happens)
                return Val(var.addr, CType(var.ctype.base, True), var.name)
            value = s.memory.read_value(var.ctype, var.addr)
            emit(s, "MEMORY_READ", addr=var.addr, size=var.ctype.size(), value=value, source=var.name)
            return Val(value, var.ctype, var.name)

        if k == "Cast":                      # reinterpret the type only; the number is unchanged
            r = yield from self._eval(n.operand)
            return Val(r.v, n.ctype, r.label)

        if k == "Un":
            if n.op == "&":
                lv = yield from self._lvalue(n.operand, "r")
                emit(s, "ADDRESS_OF", name=lv.label, addr=lv.addr)
                return Val(lv.addr, CType(lv.ctype.base, True), "&" + lv.label)
            x = yield from self._eval(n.operand)
            result = -x.v if n.op == "-" else int(not x.v)
            emit(s, "UNARY_OPERATION", operator=n.op, operand={"value": x.v, "source": x.label}, result=result)
            return Val(result, INT, str(result))

        if k == "Bin":
            op = n.op
            if op in ("&&", "||"):           # short-circuit: the right side may not run at all
                left = yield from self._eval(n.left)
                if (not left.v) if op == "&&" else left.v:
                    return Val(int(bool(left.v)), INT)
                right = yield from self._eval(n.right)
                return Val(int(bool(right.v)), INT)
            left = yield from self._eval(n.left)
            right = yield from self._eval(n.right)
            rv = right.v
            if left.ctype.ptr and op in "+-" and not right.ctype.ptr:
                rv = right.v * left.ctype.pointee().size()   # pointer arithmetic scales by element size
            result = OPERATORS[op](left.v, rv)
            ctype = INT
            if op in ("+", "-") and (left.ctype.ptr or right.ctype.ptr):
                ctype = left.ctype if left.ctype.ptr else right.ctype
            if op in _ARITHMETIC:
                # Same arithmetic, three meanings. Only BINARY_OPERATION is a VALUE computation the UI animates.
                kind = ("INDEX_ARITHMETIC" if s.index_depth else
                        "CONDITION_ARITHMETIC" if s.cond_depth else "BINARY_OPERATION")
                emit(s, kind, operator=op, left={"value": left.v, "source": left.label},
                     right={"value": right.v, "source": right.label}, result=result)
            return Val(result, ctype, str(result))

        if k == "Assign":
            # Order: find the target address, evaluate the right side, [step], write, [step].
            lv = yield from self._lvalue(n.target, "w")
            r = yield from self._eval(n.value)
            mark_destination(s, lv.label)
            if s.events and not s.quiet:
                yield                        # step 1: "compute" (reads/arithmetic just happened)
            old = s.memory.read_value(lv.ctype, lv.addr)
            s.memory.write_value(lv.ctype, lv.addr, r.v)
            new = s.memory.read_value(lv.ctype, lv.addr)   # re-read: shows wrap-around / truncation
            emit(s, "MEMORY_WRITE", addr=lv.addr, size=lv.ctype.size(), value=new, target=lv.label, **{"from": r.label})
            if lv.var is not None:
                emit(s, "VARIABLE_UPDATE", name=lv.var.name, old=old, value=new, ptr=lv.ctype.ptr)
                self._after_write(lv.var)
            if not s.quiet:
                yield                        # step 2: the write itself
            return Val(new, lv.ctype, lv.label)

        if k == "IncDec":
            lv = yield from self._lvalue(n.operand, "w")
            old = s.memory.read_value(lv.ctype, lv.addr)
            delta = (lv.ctype.pointee().size() if lv.ctype.ptr else 1) * (1 if n.op == "++" else -1)
            emit(s, "MEMORY_READ", addr=lv.addr, size=lv.ctype.size(), value=old, source=lv.label)
            emit(s, "BINARY_OPERATION", operator="+" if delta > 0 else "-",
                 left={"value": old, "source": lv.label},
                 right={"value": abs(delta), "source": str(abs(delta))}, result=old + delta, destination=lv.label)
            s.memory.write_value(lv.ctype, lv.addr, old + delta)
            new = s.memory.read_value(lv.ctype, lv.addr)
            emit(s, "MEMORY_WRITE", addr=lv.addr, size=lv.ctype.size(), value=new, target=lv.label, **{"from": ""})
            if lv.var is not None:
                emit(s, "VARIABLE_UPDATE", name=lv.var.name, old=old, value=new, op=n.op, ptr=lv.ctype.ptr)
                self._after_write(lv.var)
            if not s.quiet:
                yield
            return Val(old if n.post else new, lv.ctype, lv.label)

        if k == "Call":
            if n.name == "printf":
                if not n.args or n.args[0].k != "Str":
                    raise MemscopeError("printf needs a format string")
                values = []
                for arg in n.args[1:]:
                    values.append((yield from self._eval(arg)))
                text = format_printf(n.args[0].v, values, s)
                s.output += text
                emit(s, "OUTPUT", text=text)
                yield
                return Val(len(text), INT)
            if n.name == "strlen":
                if len(n.args) != 1:
                    raise MemscopeError("strlen expects one char pointer or array")
                pointer = yield from self._eval(n.args[0])
                return Val(len(self._string_value(pointer)), INT, "strlen")
            if n.name == "strcmp":
                if len(n.args) != 2:
                    raise MemscopeError("strcmp expects two char pointers or arrays")
                left = yield from self._eval(n.args[0])
                right = yield from self._eval(n.args[1])
                lv, rv = self._string_value(left), self._string_value(right)
                return Val(0 if lv == rv else (-1 if lv < rv else 1), INT, "strcmp")
            if n.name == "scanf":
                if len(n.args) != 2 or n.args[0].k != "Str" or n.args[0].v not in ("%d", "%s"):
                    raise MemscopeError("scanf supports scanf(\"%d\", &integer) and scanf(\"%s\", char_array)")
                # Resolve the destination through the normal expression evaluator so
                # &n, array elements, and pointer-compatible addresses use ProgramState.
                destination_expr = n.args[1]
                pointer = yield from self._eval(destination_expr)
                if not pointer.ctype.ptr:
                    raise MemscopeError("scanf destination must be an address such as &n")
                address = pointer.v
                ctype = pointer.ctype.pointee()
                target = source_text(destination_expr)
                if target.startswith("&"):
                    target = target[1:]
                variable = next((v for v in reversed(s.variables)
                                 if v.addr == address), None)
                fmt = n.args[0].v
                if fmt == "%d" and (ctype.ptr or ctype.base != "int"):
                    raise MemscopeError("scanf %d destination must be an int")
                if fmt == "%s" and (ctype.base != "char" or not variable or not variable.is_array):
                    raise MemscopeError("scanf %s destination must be a char array")
                if fmt == "%s":
                    while True:
                        while s.input_cursor >= len(s.input_values):
                            s.execution_state = "WAITING_FOR_INPUT"
                            s.waiting_for_input = {"format": "%s", "target": target, "address": address}
                            emit(s, "INPUT_REQUEST", format="%s", destination=target, address=address)
                            yield
                        raw = str(s.input_values[s.input_cursor])
                        if raw and "\x00" not in raw:
                            break
                        s.execution_state = "WAITING_FOR_INPUT"
                        s.waiting_for_input = {"format": "%s", "target": target, "address": address,
                                               "error": "Please enter a non-empty string."}
                        emit(s, "INPUT_REQUEST", format="%s", destination=target,
                             address=address, error="Please enter a non-empty string.")
                        yield
                    encoded = decode_c_escapes(raw).encode("latin1")
                    if len(encoded) + 1 > variable.count:
                        s.execution_state = "WAITING_FOR_INPUT"
                        s.waiting_for_input = {"format": "%s", "target": target, "address": address,
                                               "error": f"String is too long for {target}."}
                        emit(s, "INPUT_REQUEST", format="%s", destination=target,
                             address=address, error=f"String is too long for {target}.")
                        yield
                        return Val(0, INT, "scanf")
                    s.input_cursor += 1
                    s.execution_state = "RUNNING"
                    s.waiting_for_input = None
                    emit(s, "INPUT_RECEIVED", format="%s", destination=target,
                         address=address, value=raw)
                    for offset, byte in enumerate(encoded + b"\0"):
                        s.memory.write_byte(address + offset, byte)
                        emit(s, "MEMORY_WRITE", addr=address + offset, size=1, value=byte,
                             target=f"{target}[{offset}]", **{"from": "input"})
                    yield
                    return Val(1, INT, "scanf")
                while True:
                    while s.input_cursor >= len(s.input_values):
                        s.execution_state = "WAITING_FOR_INPUT"
                        s.waiting_for_input = {"format": "%d", "target": target, "address": address}
                        emit(s, "INPUT_REQUEST", format="%d", destination=target,
                             address=address)
                        # The replay driver stops at this yield. A later browser request
                        # supplies another input queue and resumes the same language path.
                        yield
                    raw = s.input_values[s.input_cursor]
                    if re.fullmatch(r"[+-]?\d+", str(raw).strip()):
                        break
                    s.execution_state = "WAITING_FOR_INPUT"
                    s.waiting_for_input = {"format": "%d", "target": target, "address": address,
                                           "error": "Please enter a whole number."}
                    emit(s, "INPUT_REQUEST", format="%d", destination=target,
                         address=address, error="Please enter a whole number.")
                    yield
                value = int(str(raw).strip())
                s.input_cursor += 1
                s.execution_state = "RUNNING"
                s.waiting_for_input = None
                emit(s, "INPUT_RECEIVED", format="%d", destination=target,
                     address=address, value=value)
                old = s.memory.read_value(ctype, address)
                s.memory.write_value(ctype, address, value)
                stored = s.memory.read_value(ctype, address)
                emit(s, "MEMORY_WRITE", addr=address, size=ctype.size(), value=stored,
                     target=target, **{"from": "input"})
                if variable is not None:
                    emit(s, "VARIABLE_UPDATE", name=variable.name, old=old, value=stored)
                    self._after_write(variable)
                yield
                return Val(1, INT, "scanf")
            args = []
            for arg in n.args:
                args.append((yield from self._eval(arg)))
            return (yield from self._call(n.name, args))

        raise MemscopeError(f"Unsupported expression {k}")


# ============================================
# 9. EXECUTION EVENTS
# ============================================
# An event is a plain dict {"type": ..., "line": ..., ...fields}. The engine REPORTS what happened;
# the UI decides how to draw it. Events never change program state, and the UI never needs to know
# which program is running.
#
#  type                    emitted when                                  the UI uses it to
#  ----------------------  --------------------------------------------  ----------------------------------
#  MEMORY_ALLOCATE         a variable gets its bytes                     (informational)
#  MEMORY_WRITE            bytes were written (init: True for initial)   flash changed bytes, fly value chip
#  MEMORY_READ             a value was read from memory                  highlight source bytes / variable row
#  VARIABLE_CREATE         a declaration finished                        fade the new row in, create frame row
#  VARIABLE_UPDATE         a named scalar was assigned / ++ / --         "x changed from a to b" text
#  REACTIVE_UPDATE         a dependent variable was recalculated         explanation text
#  POINTER_CREATE          a pointer variable got its first target       (arrow is drawn from state)
#  POINTER_DEREFERENCE     `*p` located its target                       select + highlight target byte
#  ADDRESS_OF              `&x` evaluated                                (informational)
#  BINARY_OPERATION        VALUE arithmetic + - * / %  (also ++ / --)    the operand -> [op] -> result stage
#  INDEX_ARITHMETIC        arithmetic inside `[...]`                     (deliberately NOT animated as a value)
#  CONDITION_ARITHMETIC    arithmetic inside an if/for/while condition   (deliberately NOT animated as a value)
#  UNARY_OPERATION         unary - or !                                  (informational)
#  ARRAY_INDEX             `b[i]` resolved to an element                 highlight the array cell
#  CONDITION_EVALUATED     a condition got a true/false                  explanation text
#  LOOP_ITERATION          a loop body is about to run (n = pass no.)    "Loop pass n" text
#  LOOP_EXIT               a loop condition became false                 explanation text
#  FUNCTION_CALL           a user function was entered                   push stack frame card
#  FUNCTION_RETURN         a user function returned                      return-value chip, pop frame
#  OUTPUT                  printf wrote text                             typewriter output panel
#  INPUT_REQUEST           scanf needs browser input                     show the input panel / pause
#  INPUT_RECEIVED          scanf accepted a value                        resume and show the write
#  ERROR                   the user's program crashed                    show the message
#
# `emit` is the ONLY function that appends to state.events.

def emit(state: ProgramState, type_: str, **fields) -> dict:
    """Record one event for the current step and return it (callers may add fields to it later, as
    POINTER_DEREFERENCE does when its value becomes known). While state.quiet > 0 (silent reactive
    re-evaluation) nothing is recorded; a throw-away dict is returned so callers need no special case."""
    event = {"type": type_, "line": state.line, **fields}
    if not state.quiet:
        state.events.append(event)
    return event


def mark_destination(state: ProgramState, label: str) -> None:
    """Attach `destination` to the most recent BINARY_OPERATION of this step, so the UI can draw
    'result -> destination'. Does nothing if there was no arithmetic or it already has a destination."""
    for e in reversed(state.events):
        if e["type"] == "BINARY_OPERATION":
            e.setdefault("destination", label)
            return


_ELEMENT_LABEL = re.compile(r"^\w+\[\d+\]$")


def explain(state: ProgramState, events: list[dict]) -> str:
    """Build the 'What just happened' sentence from the step's events ONLY (never from the program
    text), so it works for any program. Pure function: reads state.memory for one byte, changes nothing."""
    parts: list[str] = []
    errors = [e for e in events if e["type"] == "ERROR"]
    if errors:
        return "\u26a0 " + errors[0]["message"]
    op = next((e for e in reversed(events) if e["type"] == "BINARY_OPERATION"), None)
    loop = next((e for e in events if e["type"] == "LOOP_ITERATION"), None)
    fmt = lambda e, x: hex(x) if e.get("ptr") else x

    def computed(name: str) -> str:
        return f"Computed {op['left']['value']} {op['operator']} {op['right']['value']} = {op['result']} and wrote it into {name}"

    for e in events:
        t = e["type"]
        if t == "VARIABLE_CREATE":
            if op and op.get("destination") == e["name"]:
                parts.append(computed(e["name"]) + f" ({e['size']} bytes at {hex(e['address'])}).")
            else:
                holding = f", holding {e['shown']}" if e["shown"] is not None else ""
                parts.append(f"Created {e['name']}: {e['size']} bytes at {hex(e['address'])}{holding}.")
        elif t == "POINTER_CREATE":
            parts.append(f"{e['name']} points at {hex(e['target'])}.")
        elif t == "FUNCTION_CALL" and not e["implicit"]:
            parts.append(f"Called {e['name']}({', '.join(map(str, e['args']))}): a new stack frame appears.")
        elif t == "FUNCTION_RETURN":
            parts.append(f"{e['name']}() returned {e['value']}; its frame is popped.")
        elif t == "LOOP_EXIT":
            parts.append(f"Loop ends: {e['text']} is false.")
        elif t == "CONDITION_EVALUATED" and e["show"]:
            parts.append(f"Checked {e['expr']}: {'true' if e['value'] else 'false'}.")
        elif t == "POINTER_DEREFERENCE":
            if "value" in e:
                byte = state.memory.read_byte(e["addr"])
                plural = "s" if e["size"] > 1 else ""
                parts.append(f"Followed {e['ptr']} to {hex(e['addr'])} and read {e['size']} byte{plural}: "
                             f"{byte:02X} = {e['value']}.")
            else:
                parts.append(f"Followed {e['ptr']} to {hex(e['addr'])}.")
        elif t == "VARIABLE_UPDATE":
            if "op" in e:
                verb = "Incremented" if e["op"] == "++" else "Decremented"
                parts.append(f"{verb} {e['name']} from {fmt(e, e['old'])} to {fmt(e, e['value'])}.")
            elif op and op.get("destination") == e["name"]:
                parts.append(computed(e["name"]) + ".")
            else:
                parts.append(f"{e['name']} changed from {fmt(e, e['old'])} to {fmt(e, e['value'])}.")
        elif t == "MEMORY_WRITE" and not e.get("init"):
            if not any(x["type"] == "VARIABLE_UPDATE" and x["name"] == e["target"] for x in events):
                verb = "Copied" if _ELEMENT_LABEL.match(e["from"] or "") else "Wrote"
                parts.append(f"{verb} {e['value']} into {e['target']} ({hex(e['addr'])}).")
        elif t == "OUTPUT":
            parts.append(f"printf printed {json.dumps(e['text'])}.")
        elif t == "INPUT_REQUEST":
            parts.append(f"Input required: {e.get('format', '%d')} for {e.get('destination', 'destination')}.")
        elif t == "INPUT_RECEIVED":
            parts.append(f"Received {e['value']} for {e.get('destination', 'destination')}.")
        elif t == "REACTIVE_UPDATE":
            parts.append(f"{e['name']} recalculated: {e['old']} \u2192 {e['value']}.")
    if not parts:   # a pure "compute" step: say what was read
        read = next((e for e in reversed(events) if e["type"] == "MEMORY_READ"), None)
        index = next((e for e in reversed(events) if e["type"] == "ARRAY_INDEX" and e["rw"] == "r"), None)
        if read and index and re.search(r"[^\d\s]", index["expr"]) and index["addr"] == read["addr"]:
            parts.append(f"Index {index['expr']} = {index['index']}, so read {read['source']}. Its value is {read['value']}.")
        elif read:
            parts.append(f"Read {read['source']}. Its value is {read['value']}.")
        elif index:
            parts.append(f"Selected {index['array']}[{index['index']}].")
        else:
            parts.append("Evaluated.")
    return (f"Loop pass {loop['n']}. " if loop else "") + " ".join(parts)


# ============================================
# 10. OUTPUT HANDLING
# ============================================
def decode_c_escapes(text: str) -> str:
    """Decode the small escape set supported by the C-subset lexer/runtime."""
    return (text.replace("\\n", "\n").replace("\\t", "\t")
                .replace("\\0", "\0").replace('\\"', '"').replace("\\'", "'")
                .replace("\\\\", "\\"))


def read_c_string(state: ProgramState, address: int, limit: int = MEM_SIZE,
                  record_events: bool = False) -> str:
    """Read a null-terminated char sequence from simulated memory."""
    chars = []
    for offset in range(limit):
        byte = state.memory.read_byte(address + offset)
        if record_events:
            emit(state, "MEMORY_READ", addr=address + offset, size=1, value=byte,
                 source=f"string[{offset}]")
        if byte == 0:
            break
        chars.append(chr(byte))
    return "".join(chars)


def format_printf(fmt: str, args: list[Val], state: Optional[ProgramState] = None) -> str:
    """Expand printf formats including decimal, character, string, hex, and percent output.
    Also the escapes \\n, \\t, \\\". Raises MemscopeError if an argument is missing.
    The caller appends the result to state.output and emits an OUTPUT event."""
    it = iter(args)

    def substitute(m: re.Match) -> str:
        spec = m.group(1)
        if spec == "%":
            return "%"
        try:
            arg = next(it)
        except StopIteration:
            raise MemscopeError("printf: missing argument") from None
        if spec == "d":
            return str(arg.v)
        if spec == "c":
            return chr(arg.v & 0xFF)
        if spec == "s":
            if state is None or not arg.ctype.ptr or arg.ctype.base != "char":
                raise MemscopeError("printf %s needs a char pointer or array")
            return read_c_string(state, arg.v, record_events=True)
        return format(arg.v & 0xFFFFFFFF, "x")

    text = re.sub(r"%([dcsx%])", substitute, fmt)
    return decode_c_escapes(text)


# ============================================
# 11. FRONTEND (served at "/")
# ============================================
# A single self-contained HTML page: layout, CSS, and JavaScript that ONLY draws.
# It calls POST /api/execute (section 12) and renders the returned ProgramState + events.
# It contains no interpreter. To change how a program RUNS, edit sections 3-10 above; to change how
# things LOOK, edit this string. JS function map:
#   step()/reset()/api()  talk to the server        rdv()/val()   decode memory bytes for display
#   render() -> rVars, rMem (byte cells), rArr (array strip), rStack, rStage (operation/copy animation),
#               rInfo (selected byte), rNow (explanation), rEd (current-line bar)
#   tick()  eases pointer arrows        present()  shows "before" state first when a step animates
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Memscope</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{--bg:#0d1320;--l:#223049;--t:#dbe4f5;--d:#8493b3;--am:#ffb347;--mi:#4ee0b5;--vi:#9b8cff;color-scheme:dark;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
*{box-sizing:border-box}
html,body{height:100%;margin:0;background:var(--bg);color:var(--t);font:14px/1.45 "Bricolage Grotesque",system-ui,sans-serif}
body{display:grid;grid-template-rows:auto 1fr auto}
.mono,code,.cell b,.cell i,.cell u,.gv,.lab u,#ed,.ig b{font-family:"JetBrains Mono",ui-monospace,monospace}
header{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:10px 16px;border-bottom:1px solid var(--l)}
.logo{font-weight:800;font-size:18px;display:flex;align-items:center;gap:8px;margin-right:auto}
.lg{display:flex}.lg i{width:9px;height:14px;border:1.5px solid var(--am);margin-left:-1.5px}.lg i:nth-child(2){background:var(--am)}
button,select{font:inherit;background:#1a2438;color:var(--t);border:1px solid var(--l);padding:6px 12px;border-radius:6px;cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed;border-color:var(--l)}
button:hover,select:hover{border-color:var(--am)}button:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--vi)}
.go{background:var(--mi);color:#00251b;border-color:transparent;font-weight:700}.on{background:var(--vi);color:#fff;border-color:transparent}
main{display:grid;grid-template-columns:minmax(260px,1fr) minmax(250px,.8fr) minmax(380px,1.6fr);min-height:0}
section{min-height:0;overflow:auto;border-right:1px solid var(--l);display:flex;flex-direction:column}
section:last-child{border:0}
h2{font:600 13px "Bricolage Grotesque";color:var(--d);margin:0;padding:12px 16px 8px;display:flex;align-items:center;gap:8px}
h2 .seg{margin-left:auto;display:flex;gap:4px}h2 .seg button{padding:2px 8px;font-size:12px}
.empty{color:var(--d);padding:8px 16px;margin:0}
#ed{position:relative;display:flex;flex:1;min-height:0;font-size:13px;line-height:22px;overflow:hidden}
#gut{width:42px;overflow:hidden;color:#4a5a80;text-align:right;padding:12px 10px 0 0;flex:none}#gin div{height:22px}
#cw{position:relative;flex:1}#lay{position:absolute;inset:0;pointer-events:none}
#hi{margin:0;padding:12px 14px;font:inherit;white-space:pre}
#ta{position:absolute;inset:0;width:100%;height:100%;padding:12px 14px;margin:0;border:0;background:transparent;color:transparent;caret-color:#fff;font:inherit;white-space:pre;resize:none;overflow:auto;tab-size:4}
#bar,#nx{position:absolute;left:0;width:4000px;height:22px;transition:top .3s cubic-bezier(.3,1.4,.5,1)}
#bar{background:rgba(255,179,71,.18);border-left:3px solid var(--am)}#nx{background:rgba(155,140,255,.1);border-left:3px dashed var(--vi)}
.k{color:#9b8cff}.s{color:#7be0a0}.n{color:#ffb347}.f{color:#6cb8ff}.c{color:#5f6e91;font-style:italic}
#bl{display:none;padding:8px 16px;overflow:auto}.blk{display:block;width:100%;text-align:left;margin:4px 0;border:0;border-left:7px solid var(--c);border-radius:4px 16px 16px 4px;background:#1a2438}
#bl h3{margin:14px 0 2px;font-size:12px;color:var(--d);font-weight:600}
#now{padding:10px 16px;border-bottom:1px solid var(--l);min-height:92px}#now code{display:block;color:var(--am);margin-bottom:4px}
#inputPanel{display:none;margin:10px 16px;padding:10px 12px;border:1.5px solid var(--am);border-radius:6px;background:#18233a}#inputPanel b{display:block;color:var(--am)}#inputPrompt{color:var(--d);font-family:"JetBrains Mono",monospace}#inputRow{display:flex;gap:8px;margin-top:8px}#inputValue{flex:1;min-width:0;background:#0d1320;color:var(--t);border:1px solid var(--l);padding:6px 8px;border-radius:4px}#inputError{color:#ff8fa3;font-size:12px;min-height:1.2em;margin-top:4px}
.var{padding:6px 16px;border-left:3px solid var(--c);margin-bottom:6px;cursor:pointer}.var.dead{opacity:.6}.var.hl{background:#18233a}
.vh{display:flex;gap:8px;align-items:baseline}.vh b{color:var(--c);font-size:16px}.vh span{color:var(--d)}.vh code{margin-left:auto;font-size:14px}
input[type=range]{width:100%;accent-color:var(--c);margin:4px 0 0}.sl{display:flex;justify-content:space-between;font-size:10px;color:var(--d)}
.fr{margin:0 16px 16px;border:1.5px solid var(--vi);padding:10px 14px}.fr.in{animation:in .45s}.fr b{display:block;color:var(--vi);margin-bottom:4px}.fr div{display:flex;justify-content:space-between;font:13px "JetBrains Mono",monospace}
#mem{position:relative;flex:1;min-height:180px;overflow:auto}#ar{position:absolute;left:0;top:0;width:100%;pointer-events:none;overflow:visible}
.row{display:flex;align-items:center;gap:14px;padding:12px 16px;border-left:3px solid var(--c);margin:8px 0;transition:background .3s,opacity .3s}
.row.new{animation:in .5s}.row.dead{opacity:.55}.row.hl{background:color-mix(in srgb,var(--c) 14%,transparent)}
@keyframes in{from{opacity:0;transform:translateY(10px) scale(.97)}}
.lab b{font-size:22px;color:var(--c);display:block;line-height:1.1}.lab span{color:var(--d)}.lab u{display:block;text-decoration:none;color:var(--d);font-size:11px}
.lab{min-width:84px}.gs{display:flex;flex-wrap:wrap;gap:12px}.grp{display:flex;flex-direction:column;align-items:center}.cells{display:flex}
.cell{width:56px;border:1.5px solid var(--c);margin-left:-1.5px;text-align:center;cursor:pointer;padding:6px 0 3px;transition:background .25s}
.cell:hover{background:rgba(255,255,255,.07)}.cell b{display:block;font-size:19px}.cell i,.cell u{display:block;font-size:10px;color:var(--d);text-decoration:none;font-style:normal}
.cell.chg{animation:fl 1.4s}@keyframes fl{0%,25%{background:rgba(78,224,181,.6)}}
.cell.rd{background:rgba(255,179,71,.3)}.cell.sel{outline:2px solid #fff;outline-offset:2px;position:relative;z-index:2}
.gv{font-size:12px;color:var(--d);margin-top:4px}.grp em{font:11px "JetBrains Mono",monospace;color:var(--d)}
.dot{margin-left:auto;width:10px;height:10px;border-radius:50%;background:var(--am);flex:none}
#info{padding:6px 16px 16px;border-top:1px solid var(--l)}#info p{margin:8px 0 0;color:var(--d);font-size:13px}
.bits{display:flex;gap:4px}.bits span{flex:1;text-align:center;padding:6px 0;font:600 16px "JetBrains Mono",monospace;transition:background .3s}.b1{background:var(--am);color:#2a1700}.b0{background:#1a2640}
.ig{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:8px;margin-top:10px}.ig u{display:block;text-decoration:none;font-size:11px;color:var(--d)}.ig b{font-size:16px}
#out{padding:10px 16px;border-top:1px solid var(--l);display:flex;gap:14px;align-items:baseline;min-height:50px}#out span{color:var(--d);font-weight:600;font-size:13px}
#ot{font:600 18px "JetBrains Mono",monospace;color:var(--mi);white-space:pre-wrap}#ot::after{content:"▌";animation:bk 1s steps(2) infinite;color:var(--d)}@keyframes bk{50%{opacity:0}}
@media(max-width:900px){body,main{display:block;height:auto}section{overflow:visible;border:0;border-bottom:1px solid var(--l)}#ed,#bl{height:300px;flex:none}#mem{overflow-x:auto}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

#rows.bin .cell{width:98px}#rows.bin .row{min-width:max-content}.cell b.bn{display:grid;grid-template-columns:repeat(8,1fr);font-size:12px;font-weight:600}.bn span{text-align:center}.bits span{min-width:0}
.sv{margin:8px 16px 12px;padding:10px 12px;border:1px solid var(--l);border-left:3px solid var(--mi);border-radius:5px;background:#121c2d}.svh{display:flex;align-items:baseline;gap:8px}.svh b{font-size:17px;color:var(--mi)}.svh span{color:var(--d);font-size:11px}.svcells{display:flex;gap:3px;margin:8px 0 3px;overflow-x:auto}.svc{min-width:42px;border:1px solid var(--mi);padding:4px 3px;text-align:center}.svc b{display:block;font-size:15px;color:var(--t)}.svc i{display:block;font:10px "JetBrains Mono",monospace;color:var(--d);font-style:normal}.svmeta{color:var(--d);font-size:11px}.svbytes{font:11px "JetBrains Mono",monospace;color:var(--am);margin-top:4px;word-break:break-all}.sv button{padding:2px 7px;font-size:11px;margin-left:auto}
.grp.stale{opacity:.4}.grp.stale .cell{border-style:dashed}.row.read{outline:1.5px solid var(--am);outline-offset:-1px}
#now small{display:block;color:var(--d);font-size:11px;margin-bottom:2px}.fm{font-size:11px;color:var(--d);margin:2px 0}
.pr{display:flex;flex-direction:column;align-items:flex-start;margin-top:6px}.pr span{border:1.5px solid var(--mi);padding:1px 8px;font:13px "JetBrains Mono",monospace;animation:mv .5s both}.pr i{color:var(--mi);font-style:normal;padding-left:10px}.pr b{color:var(--mi)}
@keyframes mv{from{opacity:0;transform:translateX(-24px)}}
.op{display:flex;align-items:center;gap:8px;padding:12px 16px;flex-wrap:wrap}.ops{display:flex;flex-direction:column;gap:8px}
.chip{border:1.5px solid var(--vi);padding:4px 10px;text-align:center;min-width:52px;animation:mv .5s both}.chip u{display:block;font-size:11px;color:var(--d);text-decoration:none}.chip b{font:600 18px "JetBrains Mono",monospace}.chip.res{border-color:var(--mi)}.chip.dst{border-color:var(--am)}
.sym{font:700 26px "JetBrains Mono",monospace;color:var(--mi);animation:in .4s both}.cv{width:40px;height:80px;stroke:var(--vi);fill:none;stroke-width:2;stroke-dasharray:60;animation:dr .7s both}@keyframes dr{from{stroke-dashoffset:60}}
.tl{margin:0;padding:0 16px 10px 36px;font-size:12px;color:var(--d)}.tl li{animation:in .4s both}
#arrv{position:relative}.av{padding:10px 16px 4px}.avh{font-size:13px;color:var(--d);margin-bottom:6px}.avh b{color:#6cb8ff;font-size:16px}.avl{margin-top:6px;font:12px "JetBrains Mono",monospace;color:var(--d)}
.abs{display:flex;gap:4px}.ab{width:54px;border:1.5px solid #6cb8ff;text-align:center;padding:6px 0 2px;position:relative;transition:background .3s}
.ab b{display:block;font:600 18px "JetBrains Mono",monospace}.ab i{display:block;font:11px "JetBrains Mono",monospace;color:var(--d);font-style:normal}.ab em{display:block;min-height:14px;font:600 10px "JetBrains Mono",monospace;color:var(--am);font-style:normal}
.ab small{position:absolute;left:0;right:0;bottom:-15px;font-size:10px;color:var(--d)}.abs{padding-bottom:16px}.ab.stale{opacity:.45;border-style:dashed}.ab.del{border-top:3px solid var(--am)}.ab.rd{background:rgba(255,179,71,.3)}.ab.dst{background:rgba(78,224,181,.25)}
.fly{position:absolute;left:0;top:0;background:var(--mi);color:#00251b;font:700 18px "JetBrains Mono",monospace;display:flex;align-items:center;justify-content:center;z-index:3;pointer-events:none}
.abs{flex-wrap:nowrap}.ab{flex:1 1 0;min-width:0;max-width:60px}.ab.del{background:rgba(255,179,71,.14);border:2px solid var(--am)}.dx{display:block;text-decoration:none;font:700 9px "JetBrains Mono",monospace;color:#2a1700;background:var(--am);margin:2px 2px 0;padding:1px 0;white-space:nowrap;overflow:hidden}</style></head><body>
<header><div class="logo"><span class="lg"><i></i><i></i><i></i></span>Memscope</div>

<button class="go" id="run">▶ Run</button><button id="pause">⏸ Pause</button><button id="step">⏭ Step</button><button id="reset">↺ Reset</button></header>
<button id="prev" aria-label="Previous execution step">&#8592; Previous</button><button id="next" aria-label="Next execution step">&#8594; Next</button><span id="stepCount" class="mono" style="color:var(--d);font-size:12px">Step 0 / 0</span>
<main>
<section><h2>Program<span class="seg"><button id="t1" class="on">Code</button><button id="t2">Blocks</button></span></h2>
<div id="ed"><div id="gut"><div id="gin"></div></div><div id="cw"><div id="lay"><div id="bar" style="display:none"></div><div id="nx"></div><pre id="hi"></pre></div><textarea id="ta" spellcheck="false" aria-label="C code"></textarea></div></div>
<div id="bl"></div></section>
<section><div id="now"></div><div id="inputPanel"><b>WAITING FOR INPUT</b><span id="inputPrompt"></span><div id="inputRow"><input id="inputValue" type="text" autocomplete="off"><button id="inputSubmit">Enter</button></div><div id="inputError"></div></div><h2>Variables</h2><div id="vars"></div><h2>Stack</h2><div id="stack"></div></section>
<section><h2>Memory<span class="mono" style="color:var(--d);font-size:11px">little-endian · 4-byte pointers</span><span class="seg" id="rp"></span></h2>
<div id="stage"></div><div id="arrv"></div><div id="stringViews"></div><div id="mem"><div id="rows"></div><svg id="ar"></svg></div><div id="info"></div></section>
</main>
<footer id="out"><span>Output</span><div id="ot"></div></footer>
<script>
const $=id=>document.getElementById(id),BASE=0x1000,COL=['#6cb8ff','#b58cff','#ff8fa3','#7be0a0','#ffd166'];
/* Keep navigation controls beside Run/Step even though the compact HTML is assembled inline. */
['prev','next','stepCount'].forEach(id=>document.querySelector('header').appendChild(document.getElementById(id)));
/* The editor starts with this text. It is just initial editor content: the user can replace it with any program,
   and Run/Step/Reset always use whatever is in the editor at that moment. */
const STARTER='int main() {\n    int a = 10;\n    int b = 20;\n    int c = a + b;\n    return 0;\n}';
const BL=[['Variables','#6cb8ff',[['set a to 130','int a = 130;'],['change a by 1','a = a + 1;'],['set b to a + 10','int b = a + 10;']]],['Pointers','#ffb347',[['create pointer to a','char *ptr = (char *)&a;'],['move pointer one step','ptr++;'],['set value at pointer to 0','*ptr = 0;']]],['Arrays','#7be0a0',[['array of 4 numbers','int nums[4] = {10, 20, 30, 40};']]],['Output','#ff8fa3',[['print a','printf("%d", a);'],['print value at pointer','printf("%d", *ptr);']]],['Control (not simulated yet)','#8493b3',[['repeat 3 times','// repeat 3 { ... }'],['if a > 100','// if a > 100 { ... }']]]];
// Decimal is the teaching-friendly default for stored values. Addresses and pointer
// targets continue to use hx(), while the selector still lets users choose DEC/HEX/BIN/ASCII.
let mem,vars,out,shown,pc,cur,frame,pf,sel,rdd,chg,nw,note,hlv,nextA,timer,ty,repr='DEC';const AR={};
const sz=t=>t.ptr||t.b==='int'?4:1,pt=t=>({b:t.b,ptr:false}),tn=t=>t.b+(t.ptr?' *':''),hx=n=>'0x'+n.toString(16),H=n=>n.toString(16).toUpperCase().padStart(2,'0');
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;'),isEx=s=>{s=s.trim();return s&&s!=='}'&&!/^(\/\/|#|int main)/.test(s)};
/* ===== ENGINE CLIENT (replaces the old in-browser engine) =====
   All program semantics now live in the Python engine (app.py sections 3-10). This file only:
   1) asks the server for "state after N steps", 2) decodes memory bytes for display, 3) draws.
   S mirrors the server's ProgramState snapshot; nothing here executes C or keeps its own variable values. */
const MEMSZ=512;
let S=null,nSteps=0,busy=false,running=false,epoch=0,E=[],RD=new Set(),vw=null,vn=null,vfr=null,vvs=null,rt=null,propv=null,showAllStrings=false;
/* Checkpoints are serialized snapshots returned by the interpreter API.  They are the
   same state the renderer uses, rather than a second UI execution model. */
let history=[],historyPos=0,inputValues=[],pendingInput=null,resumeAfterInput=false;
const emptyState=()=>({mem:new Uint8Array(MEMSZ),vars:[],frames:[],call_stack:[],next_frame_id:0,node_visits:0,input_cursor:0,execution_state:'RUNNING',waiting_for_input:null,out:'',line:0,sp:BASE,done:false,ok:false,E:[]});
/* Decode helpers: turn the authoritative bytes in `mem` into numbers for display (little-endian). */
const readU=(a,n)=>{let v=0;for(let i=n-1;i>=0;i--)v=v*256+mem[a-BASE+i];return v};
const rdv=(t,a)=>t.ptr?readU(a,4):t.b==='char'?(x=>x>127?x-256:x)(readU(a,1)):(u=>u>=2**31?u-2**32:u)(readU(a,4));
const charText=v=>{let s='';for(let k=0;k<v.cnt;k++){const n=rdv(v.t,v.a+k);if(n===0)break;s+=String.fromCharCode(n&255)}return s};
const val=v=>v.arr&&v.t.b==='char'?'"'+esc(charText(v))+'"':v.arr?'{'+Array.from({length:v.cnt},(_,k)=>rdv(v.t,v.a+k*sz(v.t))).join(', ')+'}':v.t.ptr?hx(rdv(v.t,v.a)):rdv(v.t,v.a);
const isMove=e=>e.type==='MEMORY_WRITE'&&/^\w+\[\d+\]$/.test(e.from||'')&&/^\w+\[\d+\]$/.test(e.target||'');
async function api(steps){const r=await fetch('/api/execute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('ta').value,steps,inputs:inputValues})});return r.json()}
function applyState(d){S={mem:Uint8Array.from(d.state.mem),vars:d.state.vars,frames:d.state.frames,call_stack:d.state.call_stack||[],next_frame_id:d.state.next_frame_id||0,node_visits:d.state.node_visits||0,input_cursor:d.state.input_cursor||0,execution_state:d.state.execution_state||'RUNNING',waiting_for_input:d.state.waiting_for_input||null,out:d.state.out,line:d.state.line,sp:d.state.sp,done:d.done,ok:true,E:d.events||[]};mem=S.mem;vars=S.vars;E=S.E}
function showInputRequest(ev){
  pendingInput=ev;$('inputPanel').style.display='block';$('inputPrompt').textContent=ev.format+' → '+(ev.target||ev.destination||'destination');
  $('inputError').textContent=ev.error||'';$('inputValue').value='';$('inputValue').focus();
}
function hideInputRequest(){pendingInput=null;$('inputPanel').style.display='none';$('inputError').textContent='';$('inputValue').value=''}
function currentInputRequest(){return S&&S.execution_state==='WAITING_FOR_INPUT'?S.waiting_for_input:null}
function checkpoint(d){return JSON.parse(JSON.stringify({state:d.state,done:d.done,events:d.events||[],explanation:d.explanation||'',changed:d.changed||[],step:d.executed||0}))}
function restoreCheckpoint(cp){
  /* Restoring replaces the authoritative renderer state wholesale. Animations are cleared
     because reverse execution is a state restore, not an attempted inverse animation. */
  fin(); applyState(cp); nSteps=cp.step; if(cp.state.input_cursor!=null)inputValues=inputValues.slice(0,cp.state.input_cursor); chg=new Set(cp.changed||[]); RD=new Set(); nw=null; propv=null;
  cur=S.line?S.line-1:null; note=cp.explanation||('Restored step '+cp.step+'.'); render(1); const request=currentInputRequest(); if(request)showInputRequest(request); else hideInputRequest(); updateHistoryControls();
}
function updateHistoryControls(){
  $('prev').disabled=historyPos<=0;
  $('next').disabled=!!(S&&S.done)||!!pendingInput;
  $('step').disabled=!!pendingInput||!!(S&&S.done);
  const known=Math.max(nSteps,...history.map(h=>h.step||0));
  $('stepCount').textContent='Step '+nSteps+' / '+known;
}
/* step(): ask for state after nSteps+1 steps, then animate the difference. Returns the step's events or null when finished. */
async function step(resolveInput=false){fin();if(busy||S.done||!S.ok||(pendingInput&&!resolveInput)){stop();return null}
 const targetSteps=pendingInput?nSteps:nSteps+1;
 const ep=epoch;busy=true;let d;try{d=await api(targetSteps)}catch(e){busy=false;stop();note='⚠ Could not reach the server: '+e.message;render(1);return null}busy=false;
 if(ep!==epoch)return null;
 if(d.error){S.done=true;note='⚠ '+d.error;stop();render(1);return null}
 if(!d.events.length){S.done=true;stop();render(1);return null}
 const b={m:mem.slice(),n:S.vars.length,fr:S.frames.slice(),vs:S.vars.slice()};
 nSteps=d.executed;applyState(d);chg=new Set(d.changed);RD=new Set();nw=null;propv=null;cur=S.line?S.line-1:null;
 history=history.slice(0,historyPos+1);history.push(checkpoint(d));historyPos++;
 E.forEach(e=>{if(e.type==='MEMORY_READ'){const o=S.vars.find(v=>e.addr>=v.a&&e.addr<v.a+sz(v.t)*v.cnt);o&&RD.add(o.n)}if(e.type==='VARIABLE_CREATE')nw=S.vars.find(v=>v.a===e.address&&v.n===e.name)});
 const dd=[...E].reverse().find(e=>e.type==='POINTER_DEREFERENCE');if(dd)sel=dd.addr;
 note=d.explanation;present(b);const request=currentInputRequest();if(request){resumeAfterInput=running;showInputRequest(request);stop()}else if(E.some(e=>e.type==='INPUT_RECEIVED'))hideInputRequest();updateHistoryControls();if(S.done)stop();return E}
async function previousStep(){
  if(busy||historyPos<=0)return;
  stop(); historyPos--; restoreCheckpoint(history[historyPos]);
}
async function nextStep(){
  if(busy||S.done)return;
  await step();
}
async function submitInput(){
  if(!pendingInput||busy)return;
  const raw=$('inputValue').value.trim();
  if(pendingInput.format==='%d'&&!/^[+-]?\d+$/.test(raw)){$('inputError').textContent='Please enter a whole number.';$('inputValue').focus();return}
  if(pendingInput.format==='%s'&&!raw){$('inputError').textContent='Please enter a string.';$('inputValue').focus();return}
  /* Keep the queue transactional: a rejected value must not remain in the
     replay input list, otherwise every retry would encounter the same bad
     value before it could reach the newly submitted one. */
  const before=inputValues.length;
  inputValues.push(raw);
  const events=await step(true); const resume=resumeAfterInput; resumeAfterInput=false;
  if(!events||events.some(e=>e.type==='INPUT_REQUEST'&&e.error)){
    // Keep the request active if the server rejected the submitted value.
    if(inputValues.length>before)inputValues.pop();
  } else if(resume){$('run').click()}
}
function stop(){running=false;$('pause').classList.remove('on')}
function fin(){if(rt){clearTimeout(rt);rt=null;vw=vn=vfr=vvs=null;render(1)}}
function present(b){const ms=E.some(e=>e.type==='BINARY_OPERATION')?2000:E.some(isMove)?1100:0;
 if(ms){vw=b.m;vn=b.n;vfr=b.fr;vvs=b.vs;render();rt=setTimeout(()=>{rt=null;vw=vn=vfr=vvs=null;render(1)},ms)}else render()}
async function reset(){stop();hideInputRequest();resumeAfterInput=false;clearTimeout(rt);rt=null;vw=vn=vfr=vvs=null;propv=null;clearInterval(ty);epoch++;nSteps=0;history=[];historyPos=0;inputValues=[];S=emptyState();mem=S.mem;vars=S.vars;E=S.E;RD=new Set();shown=0;cur=null;pf=0;sel=nw=hlv=null;chg=new Set();for(const k in AR)delete AR[k];
 note='Press Step or Run. Each action the program takes changes memory in a way you can see.';render();
 try{const d=await api(0);if(d.error){S.done=true;note='⚠ '+d.error}else{applyState(d);history=[checkpoint(d)];historyPos=0}}catch(e){S.done=true;note='⚠ Could not reach the server: '+e.message}render();updateHistoryControls()}
function hl(s){return esc(s).replace(/(\/\/.*)|("(?:[^"\\]|\\.)*")|\b(int|char|return|for)\b|\b(\d+)\b|\b(printf|main)\b/g,(m,c,q,k,n,f)=>`<span class="${c?'c':q?'s':k?'k':n?'n':'f'}">${m}</span>`)}
const view=f=>{const M=mem;if(vw)mem=vw;try{return f()}finally{mem=M}},VV=()=>vvs||vars;
const lg=v=>{const x=VV().find(x=>x.n==='n'&&!x.arr);return x?Math.max(0,Math.min(v.cnt,rdv(x.t,x.a))):v.cnt};
const bits8=b=>[...b.toString(2).padStart(8,'0')].map(c=>`<span>${c}</span>`).join('');
const cellTxt=b=>repr==='HEX'?H(b):repr==='DEC'?b:repr==='BIN'?bits8(b):b>31&&b<127?esc(String.fromCharCode(b)):b?'·':'∅';
const isStringVar=v=>v.arr&&v.t.b==='char';
/* Keep the logical string view compact; the raw byte grid remains available
   through the toggle for users who need the complete array capacity. */
function stringView(v){
 const bytes=[];for(let k=0;k<v.cnt;k++){const n=rdv(v.t,v.a+k);bytes.push(n);if(n===0)break}
 const term=bytes.indexOf(0),active=term<0?bytes:bytes.slice(0,term);
 const cells=bytes.map((n,k)=>`<div class="svc"><b>${n===0?'\\0':esc(String.fromCharCode(n&255))}</b><i>${k}</i></div>`).join('');
 const hex=bytes.map(n=>H(n)).join(' '),text=active.map(n=>String.fromCharCode(n&255)).join('');
 return `<div class="sv"><div class="svh"><b>${esc(v.n)}</b><span>char[${v.cnt}]</span><button data-string-toggle="1">${showAllStrings?'Hide':'Show all '+v.cnt+' bytes'}</button></div><div class="svcells">${cells||'<span class="svmeta">(empty)</span>'}</div><div class="svmeta">String: &quot;${esc(text)}&quot; · Length: ${active.length} · Capacity: ${v.cnt} · Terminator: ${term<0?'not found':'index '+term}</div><div class="svbytes">Active bytes (hex): ${hex||'(none)'}</div></div>`;
}
function rMem(){view(()=>{const ra=new Set();E.forEach(e=>{if(e.type==='MEMORY_READ')for(let j=0;j<e.size;j++)ra.add(e.addr+j)});let h='';
 $('stringViews').innerHTML=VV().filter(isStringVar).map(stringView).join('');
 VV().forEach((v,i)=>{if(isStringVar(v)&&!showAllStrings)return;const es=sz(v.t),ln=v.arr?lg(v):1e9;let gs='';
  for(let k=0;k<v.cnt;k++){const a=v.a+k*es;let cs='';for(let j=0;j<es;j++){const b=a+j;cs+=`<div class="cell${chg.has(b)?' chg':''}${sel===b?' sel':''}${ra.has(b)?' rd':''}" data-a="${b}"><b class="${repr==='BIN'?'bn':''}">${cellTxt(mem[b-BASE])}</b><i>+${k*es+j}</i><u>${hx(b)}</u></div>`}
   gs+=`<div class="grp${k>=ln?' stale':''}">${v.arr?`<em>[${k}]${k>=ln?' stale':''}</em>`:''}<div class="cells">${cs}</div><span class="gv">= ${v.t.ptr?hx(rdv(v.t,a)):rdv(v.t,a)}</span></div>`}
  h+=`<div class="row${v===nw?' new':''}${v.dead?' dead':''}${hlv===v.n?' hl':''}${RD.has(v.n)?' read':''}" style="--c:${COL[i%5]}"><div class="lab"><b>${v.n}</b><span>${tn(v.t)}${v.arr?'['+v.cnt+']':''}</span><u>${hx(v.a)}</u></div><div class="gs">${gs}</div>${v.t.ptr?`<s class="dot" data-d="${v.n}"></s>`:''}</div>`});
 $('rows').className=repr==='BIN'?'bin':'';$('rows').innerHTML=h||'<p class="empty">Nothing in memory yet. Step the program to allocate a variable.</p>'})}
function rInfo(){if(sel==null){$('info').innerHTML='<p>Click any byte to see how the same bits can be read.</p>';return}
 const b=mem[sel-BASE],bits=b.toString(2).padStart(8,'0'),it=[['hex','0x'+H(b)],['decimal',b],['signed int8',b>127?b-256:b],['unsigned int8',b],['ascii',b>31&&b<127?"'"+String.fromCharCode(b)+"'":b?'control':'NUL']];
 $('info').innerHTML=`<h2 style="padding:8px 0">Byte at ${hx(sel)}</h2><div class="bits">${[...bits].map(x=>`<span class="b${x}">${x}</span>`).join('')}</div><div class="ig">${it.map(([k,v])=>`<div><u>${k}</u><b>${v}</b></div>`).join('')}</div><p>The eight bits never change between these readings. Only the interpretation does.</p>`}
function rVars(){view(()=>{$('vars').innerHTML=VV().map((v,i)=>`<div class="var${v.dead?' dead':''}${hlv===v.n?' hl':''}" data-n="${v.n}" style="--c:${COL[i%5]}"><div class="vh"><b>${v.n}</b><span>${tn(v.t)}${v.arr?'['+v.cnt+']':''}</span><code data-v="${v.n}">${val(v)}</code></div>${v.expr?`<div class="fm">= ${esc(v.expr)} · follows ${v.deps.join(', ')}</div>`:''}</div>`).join('')||'<p class="empty">No variables yet.</p>'})}
function rStack(){view(()=>{const F=vfr||S.frames,V=VV();$('stack').innerHTML=F.length?F.map((f,k)=>`<div class="fr${k>=pf?' in':''}"><b>${f.fn}()</b>${V.filter(v=>v.frame===f.id&&(vfr||!v.dead)).map(v=>`<div><span>${v.n}</span><span>= ${val(v)}</span></div>`).join('')||'<div><span>(empty)</span></div>'}</div>`).reverse().join(''):'<p class="empty">No active frames.</p>';pf=F.length})}
function rEd(){const L=$('ta').value.split('\n'),b=$('bar'),n=$('nx');b.style.display=cur==null?'none':'block';b.style.top=12+(cur||0)*22+'px';n.style.display='none'}
function rNow(){const L=$('ta').value.split('\n');$('now').innerHTML='<small>What just happened</small>'+(cur==null||propv?'':`<code>line ${cur+1}: ${esc(L[cur].trim())}</code>`)+esc(note)+(propv?`<div class="pr">${propv.map((p,k)=>`${k?'<i>↓</i>':''}<span style="animation-delay:${k*.4}s"><b>${p.n}</b> ${p.o} → ${p.v}</span>`).join('')}</div>`:'')}
function rArr(){view(()=>{const V=VV(),tg=V.filter(x=>!x.arr&&!x.t.ptr&&['i','j','k'].includes(x.n)).map(x=>[x.n,rdv(x.t,x.a)]),dv=V.find(x=>x.n==='deleteIndex');
 $('arrv').innerHTML=V.filter(v=>v.arr).map(v=>{const n=lg(v);let b='';
  for(let k=0;k<v.cnt;k++)b+=`<div class="ab${k>=n?' stale':''}${dv&&rdv(dv.t,dv.a)===k?' del':''}" data-k="${v.n}:${k}"><b>${rdv(v.t,v.a+k*sz(v.t))}</b><i>${k}</i><em>${tg.filter(t=>t[1]===k).map(t=>t[0]).join(' ')}</em>${dv&&rdv(dv.t,dv.a)===k&&k<n?'<u class="dx">✕ delete</u>':''}${k>=n?'<small>stale</small>':''}</div>`;
  const live=Array.from({length:n},(_,k)=>rdv(v.t,v.a+k*sz(v.t)));
  return`<div class="av"><div class="avh"><b>${v.n}</b> logical length ${n}${n<v.cnt?` of ${v.cnt} slots`:''}${dv?` · deleteIndex = ${rdv(dv.t,dv.a)}`:''}</div><div class="abs">${b}</div><div class="avl">logical array: [${live.join(', ')}]</div></div>`}).join('')})}
function rStage(){const st=$('stage'),A=$('arrv'),b=E.filter(e=>e.type==='BINARY_OPERATION').pop(),fr=E.find(e=>e.type==='FUNCTION_RETURN'),ix=E.filter(e=>e.type==='ARRAY_INDEX'&&e.rw==='r'&&/[^\d\s]/.test(e.expr)).pop(),xr=ix&&E.filter(e=>e.type==='MEMORY_READ'&&e.addr===ix.addr).pop();st.innerHTML='';
 if(b){const u=Math.floor((((b.result%2**32)+2**32)%2**32)),by=[0,1,2,3].map(i=>H(Math.floor(u/2**(8*i))%256)).join(' '),num=l=>/^\d+$/.test(l),dn=b.destination,nw2=E.some(e=>e.type==='VARIABLE_CREATE'&&e.name===dn),L=b.left.source,R=b.right.source,
  ch=(l,x,d,c='')=>`<div class="chip ${c}" style="animation-delay:${d}s"><u>${num(l)?'value':l}</u><b>${x}</b></div>`,
  tl=[`${num(L)?'Use':'Read'} ${L} → ${b.left.value}`,`${num(R)?'Use':'Read'} ${R} → ${b.right.value}`,`Perform ${b.left.value} ${b.operator} ${b.right.value}`,`Produce ${b.result}`,dn?`${nw2?'Allocate and write':'Write'} ${b.result} into ${dn}${/^\w+$/.test(dn)?': '+by:''}`:`Hold ${b.result}`,'Update the memory view'];
  st.innerHTML=`<div class="op"><div class="ops">${ch(L,b.left.value,0)}${ch(R,b.right.value,.3)}</div><svg class="cv" viewBox="0 0 40 100"><path d="M0 25L40 50M0 75L40 50"/></svg><div class="sym" style="animation-delay:.7s">${b.operator}</div>${ch('result',b.result,1.1,'res')}${dn?`<span>→</span>${ch(dn,b.result,1.5,'dst')}`:''}</div><ol class="tl">${tl.map((x,k)=>`<li style="animation-delay:${.3*k+.1}s">${esc(x)}</li>`).join('')}</ol>`}
 else if(ix&&xr)st.innerHTML=`<div class="op"><div class="chip"><u>index</u><b>${esc(ix.expr)} = ${ix.index}</b></div><span>locates →</span><div class="chip res" style="animation-delay:.4s"><u>${esc(xr.source)}</u><b>${xr.value}</b></div><span style="animation:mv .5s .8s both">value read ✓</span></div>`;else if(fr)st.innerHTML=`<div class="op"><div class="chip res"><u>${fr.name}() returns</u><b>${fr.value}</b></div><span>→ back to the caller</span></div>`;
 const q=k=>A.querySelector(`[data-k="${k}"]`);E.forEach(e=>{if(e.type==='ARRAY_INDEX'){const x=q(e.array+':'+e.index);x&&x.classList.add(e.rw==='w'?'dst':'rd')}});
 const mv=E.find(isMove);if(mv){const f=x=>x.replace(/\[(\d+)\]/,':$1'),s=q(f(mv.from)),d=q(f(mv.target));
  if(s&&d){s.classList.add('rd');d.classList.add('dst');const r=A.getBoundingClientRect(),a=s.getBoundingClientRect(),c=d.getBoundingClientRect(),y=document.createElement('div');y.className='fly';y.textContent=mv.value;y.style.cssText=`width:${a.width}px;height:${a.height}px`;A.appendChild(y);
   y.animate([{transform:`translate(${a.left-r.left}px,${a.top-r.top}px)`},{transform:`translate(${c.left-r.left}px,${c.top-r.top}px)`}],{duration:850,delay:100,easing:'ease-in-out',fill:'both'})}}}
function render(f){rVars();rMem();rArr();if(!f)rStage();rInfo();rStack();rEd();rNow();if(S.out.length>shown)typeOut();ro()}
function ro(){$('ot').textContent=S.out.slice(0,shown)}
function typeOut(){clearInterval(ty);ty=setInterval(()=>{if(shown<S.out.length){shown++;ro()}else clearInterval(ty)},80)}
function tick(){const m=$('mem'),mr=m.getBoundingClientRect(),ox=mr.left-m.scrollLeft,oy=mr.top-m.scrollTop;let p='';$('ar').setAttribute('height',m.scrollHeight);
 vars.forEach((v,i)=>{if(!v.t.ptr)return;let a;try{a=rdv(v.t,v.a)}catch(e){return}const d=document.querySelector(`[data-d="${v.n}"]`),c=document.querySelector(`#rows [data-a="${a}"]`);if(!d||!c)return;
  const r=d.getBoundingClientRect(),q=c.getBoundingClientRect(),sx=r.left-ox+5,sy=r.top-oy+5,above=sy<q.top-oy,tx=q.left-ox+q.width/2,ty2=above?q.top-oy-2:q.bottom-oy+2;
  const s=AR[v.n]||(AR[v.n]={x:sx,y:sy});s.x+=(tx-s.x)*.16;s.y+=(ty2-s.y)*.16;const dy=above?1:-1,col=COL[i%5];
  p+=`<path d="M${sx},${sy} C${sx+70},${sy} ${s.x},${s.y-dy*70} ${s.x},${s.y}" fill="none" stroke="${col}" stroke-width="2.5"/><path d="M${s.x},${s.y} l-6,${-dy*11} h12z" fill="${col}"/>`});
 $('ar').innerHTML=p;requestAnimationFrame(tick)}
function sync(){const t=$('ta');$('hi').innerHTML=hl(t.value)+'\n';$('gin').innerHTML=t.value.split('\n').map((_,i)=>`<div>${i+1}</div>`).join('')}
function ins(code){const t=$('ta'),L=t.value.split('\n'),m=/main\(\)/.test(t.value);let i=L.findIndex(s=>/^\s*return\b/.test(s));if(i<0)i=L.length;
 if(i>0&&!L[i-1].trim()&&i<L.length)i--;L.splice(i,0,(m?'    ':'')+code);t.value=L.join('\n');sync();reset();tab(1)}
function tab(c){$('ed').style.display=c?'flex':'none';$('bl').style.display=c?'none':'block';$('t1').className=c?'on':'';$('t2').className=c?'':'on'}
$('ta').oninput=()=>{sync();reset()};
$('ta').onscroll=e=>{const t=e.target;$('lay').style.transform=`translate(${-t.scrollLeft}px,${-t.scrollTop}px)`;$('gin').style.transform=`translateY(${-t.scrollTop}px)`};
/* Lightweight VS Code-style pairing for the existing textarea editor. The textarea remains
   the real source buffer; these edits only adjust its text and selection, then use the same
   sync/reset path as ordinary typing. */
$('ta').onkeydown=e=>{
 const t=e.currentTarget, pairs={'(':')','[':']','{':'}','"':'"',"'":"'"};
 const opens=Object.keys(pairs), closes=Object.values(pairs), start=t.selectionStart, end=t.selectionEnd, text=t.value;
 if(e.ctrlKey||e.metaKey||e.altKey)return;
 const replace=(value,caret)=>{e.preventDefault();t.value=value;t.selectionStart=t.selectionEnd=caret;sync();reset()};
 if(closes.includes(e.key)&&start===end&&text[start]===e.key){
   // The close character was inserted by its opener; move over it instead of duplicating it.
   e.preventDefault();t.selectionStart=t.selectionEnd=start+1;return;
 }
 if(opens.includes(e.key)){
   const close=pairs[e.key];
   replace(text.slice(0,start)+e.key+close+text.slice(end),start+1);
   return;
 }
 if(e.key==='Backspace'&&start===end&&start>0&&start<text.length&&pairs[text[start-1]]===text[start]){
   // Delete an untouched empty pair as one unit.
   replace(text.slice(0,start-1)+text.slice(start+1),start-1);return;
 }
 if(e.key==='Enter'&&start===end&&text[start-1]==='{'&&text[start]==='}'){
   const lineStart=text.lastIndexOf('\n',start-1)+1;
   const indent=(text.slice(lineStart,start).match(/^[ \t]*/) || [''])[0];
   const inner='\n'+indent+'    '+'\n'+indent;
   replace(text.slice(0,start)+inner+text.slice(start),start+1+indent.length+4);return;
 }
};
$('step').onclick=()=>{stop();step()};$('reset').onclick=reset;
$('prev').onclick=previousStep;$('next').onclick=nextStep;
$('inputSubmit').onclick=submitInput;
$('inputValue').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();submitInput()}};
$('pause').onclick=()=>stop();
$('run').onclick=async()=>{if(running)return;if(S.done)await reset();running=true;$('pause').classList.add('on');
 while(running){const e=await step();if(!e||S.done||e.some(x=>x.type==='INPUT_REQUEST'))break;await new Promise(r=>setTimeout(r,e.some(x=>x.type==='BINARY_OPERATION'||isMove(x))?2400:1100))}stop()};
$('rp').innerHTML=['HEX','DEC','BIN','ASCII'].map(r=>`<button data-r="${r}" class="${r===repr?'on':''}">${r}</button>`).join('');
$('rp').onclick=e=>{const r=e.target.dataset.r;if(!r)return;repr=r;[...$('rp').children].forEach(b=>b.className=b.dataset.r===r?'on':'');rMem()};
$('rows').onclick=e=>{const c=e.target.closest('.cell');if(c){sel=+c.dataset.a;rMem();rInfo()}};
$('stringViews').onclick=e=>{if(e.target.closest('[data-string-toggle]')){showAllStrings=!showAllStrings;rMem()}};
$('vars').onclick=e=>{if(e.target.type==='range')return;const v=e.target.closest('.var');if(v){hlv=hlv===v.dataset.n?null:v.dataset.n;rVars();rMem()}};
$('bl').innerHTML=BL.map(([h,c,bs])=>`<h3>${h}</h3>`+bs.map(([l,s])=>`<button class="blk" style="--c:${c}" data-c="${esc(s).replace(/"/g,'&quot;')}">${l}</button>`).join('')).join('');
$('bl').onclick=e=>{const c=e.target.dataset.c;if(c)ins(c.replace(/&lt;/g,'<').replace(/&amp;/g,'&'))};
$('t1').onclick=()=>tab(1);$('t2').onclick=()=>tab(0);
/* Keyboard controls intentionally ignore editable targets so arrows, space, and R
   retain their normal meaning while the source editor or a form field has focus. */
document.addEventListener('keydown',e=>{
 const el=e.target,editing=el&&(['INPUT','TEXTAREA','SELECT'].includes(el.tagName)||el.isContentEditable);
 if(editing)return;
 if(e.key==='ArrowRight'){e.preventDefault();nextStep()}
 else if(e.key==='ArrowLeft'){e.preventDefault();previousStep()}
 else if(e.key===' '){e.preventDefault();if(running)stop();else $('run').click()}
 else if(e.key==='r'||e.key==='R'){e.preventDefault();reset()}
});
$('ta').value=STARTER;sync();reset();tick();
</script></body></html>
'''


# ============================================
# 12. API ROUTES
# ============================================
app = Flask(__name__)     # Vercel looks for a module-level variable called `app`.


def run_program(source: str, steps: int, inputs: Optional[list[str]] = None) -> dict:
    """Parse `source`, replay it for `steps` steps from the beginning, and describe the result.

    Why replay: the server keeps no session. Replaying is deterministic and cheap for teaching-size
    programs (bounded by MAX_STEPS / MAX_NODE_VISITS).

    Returns a JSON-ready dict:
      executed     how many steps actually ran (less than `steps` => the program finished earlier)
      done         the program finished or crashed
      events       events of the LAST executed step; [] if the program had already finished before `steps`
      explanation  the 'What just happened' sentence for that step
      changed      addresses of bytes that differ from before that step
      state        the authoritative ProgramState snapshot: mem (all bytes), vars, frames, out, line, sp
    Raises MemscopeError for syntax errors (reported to the user)."""
    if len(source) > MAX_SOURCE_CHARS:
        raise MemscopeError("Program is too long")
    program = Parser(tokenize(source)).parse_program()
    interp = Interpreter(program, inputs or [])
    state = interp.state
    executed, events, changed = 0, [], []
    for _ in range(min(steps, MAX_STEPS)):
        before = bytes(state.memory.data)
        step_events = interp.step()
        if step_events is None:   # program already finished: report "no events" so the client knows it is done
            events, changed = [], []
            break
        executed, events = executed + 1, step_events
        changed = [MEM_BASE + i for i, (a, b) in enumerate(zip(before, state.memory.data)) if a != b]
        # INPUT_REQUEST is a real execution boundary. Do not replay beyond scanf
        # until the browser supplies a value in a subsequent request.
        if any(e.get("type") == "INPUT_REQUEST" for e in step_events):
            break
    return {
        "executed": executed,
        "done": state.done,
        "events": events,
        "explanation": explain(state, events) if events else "",
        "changed": changed,
        "state": {
            "mem": list(state.memory.data),
            "vars": [v.to_json() for v in state.variables],
            "frames": [{"id": f.id, "fn": f.fn} for f in state.frames],
            # These fields keep checkpoints faithful to the interpreter state.  The browser
            # does not execute with them, but restoring them makes navigation complete and
            # leaves ProgramState as the only source of truth.
            "call_stack": list(state.call_stack),
            "next_frame_id": state.next_frame_id,
            "node_visits": state.node_visits,
            "input_cursor": state.input_cursor,
            "execution_state": state.execution_state,
            "waiting_for_input": state.waiting_for_input,
            "out": state.output,
            "line": state.line,
            "sp": state.memory.sp,
        },
    }


@app.get("/")
def index() -> Response:
    """Serve the single-page frontend (section 11)."""
    return Response(INDEX_HTML, mimetype="text/html")


@app.post("/api/execute")
def api_execute():
    """Body: {"source": "<C code>", "steps": N, "inputs": [...]}. Response: see run_program().
    User-program errors (syntax errors) come back as {"error": "..."} with HTTP 200 because they are
    expected input problems, not server failures."""
    data = request.get_json(silent=True) or {}
    source, steps, inputs = data.get("source"), data.get("steps", 0), data.get("inputs", [])
    if not isinstance(source, str) or not isinstance(steps, int) or steps < 0:
        return jsonify({"error": "Expected JSON like {\"source\": \"...\", \"steps\": 3}"}), 400
    if not isinstance(inputs, list) or any(not isinstance(x, (str, int)) for x in inputs):
        return jsonify({"error": "inputs must be a list of submitted values"}), 400
    try:
        return jsonify(run_program(source, steps, [str(x) for x in inputs]))
    except MemscopeError as err:
        return jsonify({"error": str(err)})


# ============================================
# 13. APPLICATION STARTUP
# ============================================
if __name__ == "__main__":
    # Local development only. On Vercel the platform imports `app` and serves it; this block never runs there.
    app.run(debug=True, port=5000)
