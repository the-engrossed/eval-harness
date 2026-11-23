# eval-harness — agent instructions

1. Read `SPEC.md` and `.cursor/rules/*.mdc` before writing code. They are the rules.
2. `make demo` must run offline with no keys, no Docker, no network.
3. Plan first for multi-file changes; wait for approval. Failing test first.
4. Never hardcode a metric, price or result. Every number comes from `runs/`.
5. Run `make check && make names` before saying a task is done, and paste the output.
