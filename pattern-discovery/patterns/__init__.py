"""Data-driven discovery of predictive price-shape patterns.

Premise (from technical analysis): recurring shapes in the price path may
precede directional moves. Rather than hand-crafting patterns (triangles, etc.)
we sweep a moving window over many series at chosen time scales, and ask which
*shapes* carry predictive information about the forward return.

Central caveat (Keogh & Lin 2005): unsupervised clustering of sliding-window
subsequences is *meaningless* -- centroids are artefacts of the sweep, roughly
data-independent. So the supervised framing (select shapes by their forward-return
edge) is the load-bearing idea; the unsupervised version is only a sanity gate.

Reuses the `mc` package (loader, returns, walk-forward scoring) from the
monte-carlo-experiment sibling project.
"""
