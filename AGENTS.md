# OpenRenderGraph contributor notes

## Comments

Write comments in plain language. A reader new to the code should understand them on the first read.

- Use short sentences and common words. One idea per sentence.
- Say what the code does and why. Name things directly: "the barriers that move the texture to CopyDest", not "what takes
  the texture on to CopyDest".
- Don't stack clauses with colons, semicolons and parentheses. Split them into sentences, or use a list for a sequence of
  steps.
- Prefer full words over abbreviations, except common ones (GPU, CPU, API, RHI).
- Keep Doxygen tags (`@brief`, `@param`, ...) where they exist.
- Don't restate the code. Explain intent, constraints and non-obvious behavior.

Example. Instead of:

```cpp
// Ahead of the caller's commands: the handoff (the batch's entry barriers, then what takes textures on from Common after a
// queue-family acquire), a full barrier for a closed execution, and the pass's own barriers before it.
```

write:

```cpp
// Before the caller's commands, record:
// - the entry barriers that hand each resource over from its previous user,
// - for textures acquired from another queue family, the barriers that move them out of Common,
// - a full memory barrier when executions are closed,
// - the pass's own barriers.
```
