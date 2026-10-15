# Memscope

Memscope is a visual, educational interpreter for a small subset of C. You write code on the left, step through it, and watch memory, variables, pointers, arrays and the call stack change. The goal is to make the invisible state of a program visible.

It is a **teaching simulator**, not a C compiler. Its simulated memory is not identical to real C memory (see [Memory model](#memory-model)).

## How it works

```
C source -> tokenize() -> Parser -> AST -> Interpreter (one Python generator yield = one step)
         -> ProgramState + events -> JSON (/api/execute) -> browser draws it
```

- **The engine runs on the server, in Python.** The browser never executes C. It asks for "the state after N steps" and draws the result.
- **`ProgramState` is the single source of truth.** Variable values are not stored anywhere except the simulated memory bytes. The UI decodes those bytes for display.
- **Events** (`MEMORY_READ`, `MEMORY_WRITE`, `BINARY_OPERATION`, ...) are generic facts about what a step did. The frontend animates event types, never specific example programs.
- **The server is stateless.** To run step N, `/api/execute` re-parses the source and replays N steps from the start. The engine is deterministic, so this always gives the same result. This is what makes it work on serverless hosting.

## Repository structure

```
memscope/
├── app.py            # everything: engine, API, and the single-page frontend
├── requirements.txt  # Flask
├── .gitignore
└── README.md
```

### Where things are in `app.py`

| Section | What it contains |
|---|---|
| 1-2 | Imports and configuration (`MEM_BASE`, `MEM_SIZE`, size constants, safety limits) |
| **3. Program state** | `CType`, `Variable`, `Frame`, `Val`, `LValue`, and **`ProgramState`** |
| **4. Memory model** | `Memory`: `read_byte`, `write_unsigned`, `read_int`, `read_value`, `allocate`, `free` |
| **5. Variables and scopes** | `lookup_variable`, `create_variable`, `pop_scope` (variable lifetime rules) |
| 6. Lexer | `tokenize()` |
| 7. Parser / AST | `Parser` (recursive descent) and the AST node shapes (see `node()` docstring) |
| **8. Execution engine** | `Interpreter`: `step()`, `_exec` (statements), `_eval` (values), `_lvalue` (addresses), `_call`, `_propagate` (reactivity) |
| **9. Execution events** | The event table, `emit()`, `explain()` ("what just happened" text) |
| 10. Output handling | `format_printf()` |
| **11. Frontend** | `INDEX_HTML`: layout, CSS and drawing JavaScript (no interpreter in it) |
| **12. API routes** | `run_program()`, `GET /`, `POST /api/execute` |
| 13. Startup | `app.run(...)` for local development |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                    # http://127.0.0.1:5000
```

## Modifying `app.py`

- **Add a language feature** (for example `else if`, `switch`, `%s`): add the AST node in the parser (section 7), handle it in `_exec` or `_eval` (section 8), and emit existing event types where possible. Do not special-case example programs or variable names.
- **Add an event type:** emit it with `emit()`, document it in the table at the top of section 9, add a sentence in `explain()` if it should appear in the explanation, and handle it in the frontend's `render*` functions.
- **Change how things look:** edit `INDEX_HTML` (section 11). The key functions are `render`, `rMem`, `rArr`, `rStage`, `rVars`, `rStack`.
- **Fix wrong values:** fix the engine (sections 4-8), not the UI. If `ProgramState` is right but the picture is wrong, fix the event or the renderer.
- **Variable lifetime:** declarations allocate once per scope entry. Loop iterations reuse the variable. `pop_scope` removes a scope's variables and rewinds the stack pointer, so re-entering a scope reuses the same addresses. See the comment block at the top of section 5.

## Memory model

| Property | Value |
|---|---|
| Addressing | Byte-addressable. Addresses are plain integers starting at `0x1000` |
| Size | 512 bytes (`0x1000` to `0x11FF`) |
| `int` | 4 bytes, little-endian, signed two's complement (`130` is stored as `82 00 00 00`) |
| `char` | 1 byte, signed |
| Pointer | 4 bytes, unsigned address, little-endian |
| Arrays | One contiguous allocation of `count x element size` bytes. `arr[i]` is at `base + i * element size` |
| Allocation | Stack-style bump pointer. Each allocation is rounded up to a multiple of 4 bytes |
| Lifetime | Leaving a scope frees its variables. The bytes are not erased, so a reused slot shows stale data until written |
| Reads and writes | Every read and write goes through `Memory.read_value` / `write_value`. An out-of-range address raises a simulated segmentation fault |

Not modelled: heap (`malloc`/`free`), structs, multi-level pointers, alignment rules or real addresses.

## Supported C subset

`int`, `char`, `void`, single-level pointers, arrays, `+ - * / %`, comparisons, `&&` `||` `!`, `&` and `*`, `[]`, `++` `--`, compound assignment, `if`/`else`, `for`, `while`, functions with parameters and `return`, and `printf` with `%d %c %x`.

## Deploy to Vercel

No `vercel.json` is needed. Vercel detects a Flask application when a module-level `app` exists in `app.py`, and installs dependencies from `requirements.txt`.

1. Push this repository to GitHub.
2. In Vercel choose **Add New -> Project**, import the repository and deploy with the default settings.
3. Or from the command line: `npm i -g vercel`, then run `vercel` in this folder.

Notes:
- The `if __name__ == "__main__"` block only runs locally.
- Each request is independent. Execution is bounded by `MAX_STEPS` and `MAX_NODE_VISITS` so an infinite loop in a user program cannot hang the function.
- Google Fonts are loaded from the internet; the page falls back to system fonts without them.
