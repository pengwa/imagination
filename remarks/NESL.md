Here's a curated set of resources on NESL / nested (data) parallelism across formats.

## Primary papers & technical reports (PDF)

- **NESL: A Nested Data-Parallel Language (Version 3.1)** — Guy Blelloch, CMU-CS-95-170, 1995. The canonical language definition with worked algorithm examples.
  - CMU page: https://www.cs.cmu.edu/~scandal/nesl.html
  - DTIC full report (PDF): https://apps.dtic.mil/sti/citations/ADA302167
  - Internet Archive mirror: https://archive.org/details/DTIC_ADA302167

- **Implementation of a Portable Nested Data-Parallel Language** — Blelloch, Chatterjee, Hardwick, Sipelstein, Zagha (JPDC 1994). Covers the VCODE/CVL implementation and benchmarks.
  - PDF: https://graphics.stanford.edu/~sjm/faculty/pubs/ (also on aminer)

- **Programming Parallel Algorithms** — Blelloch, *CACM* Mar 1996. The most readable motivation for nested data parallelism + the work/depth cost model.
  - https://cseweb.ucsd.edu/~ccalvert/... (also linked from the CMU NESL page)

- **Harnessing the Multicores: Nested Data Parallelism in Haskell** — Peyton Jones, Leshchinskiy, Keller, Chakravarty (FSTTCS 2008). Modern successor (Data Parallel Haskell), focuses on the vectorisation/flattening transform.
  - Microsoft Research (paper + slides): https://www.microsoft.com/en-us/research/publication/harnessing-the-multicores-nested-data-parallelism-in-haskell/
  - Springer LNCS: https://link.springer.com/chapter/10.1007/978-3-540-89330-1_10

## Slides / PPT

- **Nested Data Parallelism in Haskell** — Simon Peyton Jones talk (Northeastern, 2010). Available in **PowerPoint and PDF** directly from MS Research (linked on the paper page above). Excellent visual intro to flat vs. nested DP and flattening.
- **"Ndp Slides"** (OSCON 2007) — 44-slide overview: https://www.slideshare.net/oscon2007/ndp-slides
- **Modern Parallel Languages** — Kathy Yelick, UC Berkeley CS294 Lecture 2 (data parallelism / NESL). PDF from her course page: https://people.eecs.berkeley.edu/~yelick/

## Course material & reference

- **CMU NESL homepage** — tutorial, quick reference, library of parallel algorithms, and an online interpreter: https://www.cs.cmu.edu/~scandal/nesl.html
- **Vector Models for Data-Parallel Computing** — Blelloch (MIT Press, 1990). The book-length foundation behind the segmented-scan/flattening ideas.
- **UNC "Nested parallelism and flattening" bibliography** — a whole reading list of flattening-transform papers: https://www.cs.unc.edu/~prins/ (Prins group)
- **HaskellWiki – Data Parallel Haskell / References**: https://wiki.haskell.org/GHC/Data_Parallel_Haskell/References

