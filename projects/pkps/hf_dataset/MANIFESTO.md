# A manifesto for reusable evaluation embeddings

This cache should make a new experiment cheaper to start and easier to audit.
Its purpose is to preserve reusable measurements, including their limitations.

1. **Preserve evidence.** Keep the vectors unchanged, link each vector to exact
   input text, and retain source revisions and checksums. A change in encoder,
   prompt, preprocessing, or token limit deserves a new version.
2. **Make identity explicit.** An embedding index identifies a row in a particular
   release's vector bank. A model ID, query ID, or identical text serves a
   different purpose. Publish the mappings rather than relying on file order.
3. **Separate features from evaluation targets.** Reusing a frozen encoder is
   useful; silently admitting held-out responses or full-depth labels changes
   the experiment. State the observation and reference-label budgets.
4. **Keep missing outcomes missing.** An absent benchmark score is not zero and
   is not a verified prediction. Report the evaluated support alongside error.
5. **Make simple comparisons easy.** The same row tables should support sample
   means, additive score models, constant response kernels, and query-aware
   methods. A result belongs to its actual information regime and cohort.
6. **Treat cache design as part of the measurement.** Which questions were asked,
   when the run occurred, and how answers were formatted can change a model's
   representation. Preserve those distinctions instead of naming every distance
   a pure measure of capability.
7. **Make extensions traceable.** Pin Hub revisions in downstream projects,
   retain the original observations, and describe new subsets or transforms.
   The 45-model cohort, 69-model graph, and style probe are related views, not
   independent replications.

The working promise is simple: a future project should be able to load the same
vectors, recover their meaning, check their integrity, and make its own claims.

`MANIFEST.json` is the machine-readable inventory that accompanies this statement;
`SHA256SUMS` verifies its payload and the manifest itself.
