# Security Policy

## Reporting a vulnerability

Email **donglinzhuang@outlook.com** with a description of the issue and, if
possible, a reproduction. Please do not open a public issue for a security
problem.

## Scope

This project compiles and launches GPU kernels. Reports of interest include:

- memory-safety or out-of-bounds access in a kernel, for example through a page
  table or sequence length the planner accepts;
- a path where caller-supplied tensors are trusted in a way that lets a malformed
  pool or page table corrupt another allocation;
- kernel-cache poisoning: the on-disk cache is keyed by a source closure, and a
  way to make it serve a kernel compiled from different source is in scope.

Out of scope:

- numerical differences within the documented accuracy envelope;
- performance regressions;
- anything that requires an attacker who already has code execution in the
  process running the kernel.
