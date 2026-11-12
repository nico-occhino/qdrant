# Open questions

1. How can logical level membership be bound atomically to Qdrant's holder swap and segment manifest across failures? Can it be derived safely at reopen?
2. How can normal optimizer selection be constrained or integrated without breaking WAL, proxy propagation and flush ordering?
3. How can a pinned query view bind level membership, model and postings, while respecting point-version visibility during updates?
4. Which reference normalization convention is intended for buffer versus learned levels, and how should uncalibrated independent routers be evaluated?
5. How should filters, fallback, cancellation and duplicate point versions work with a global bucket-selection/scoring API?
6. What do the thesis diagrams and remaining reference persistence source establish beyond the inspected text?
7. How can a test-only holder topology with actual LMI segments and deterministic interleavings be built, then extended to collection-level reopen and failure boundaries?
