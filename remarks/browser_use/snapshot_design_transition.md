Detailed Commit Analysis
PR #2705: "Generate python highlights for browser screenshot" (Aug 17-30, 2025)
This is the mega-PR (20 commits, 624 additions) where everything started.

Commit 6ad7e70 — "Add Python-based screenshot highlighting with element bounding boxes"
What changed: Created python_highlights.py from scratch. Restructured dom_watchdog.py to run DOM building and screenshot capture in parallel via asyncio.create_task().
Architecture: Takes a clean CDP screenshot → builds DOM tree (in parallel) → draws PIL bounding boxes on the screenshot → sends annotated screenshot to LLM.
Screenshots sent to LLM: 1 annotated screenshot per step (with bounding boxes + index labels drawn on it via PIL). The system prompt's <browser_vision> described this.
Motivation: Replace JavaScript-based DOM highlighting, which mutated the page and caused timing issues, flicker, and side-effects.
Key comment from PR: "Replaced JavaScript-based element highlights with fast Python image processing on screenshots. DOM build and screenshot now run in parallel for lower latency and no in-page side effects."
Commit e8bd858 — "Add filter for highlight IDs in screenshot generation"
What changed: Added filter_highlight_ids to BrowserProfile (default True). When enabled, index labels are only shown on elements with < 5 characters of text content. Elements with enough text (like "Submit", "Search") don't get a numeric label overlay.
Motivation: This is directly addressing the problem you described — index numbers being confused with page content. If a button says "Submit", the LLM sees the text in the DOM tree and doesn't need [42] painted on the screenshot. Only tiny/unlabeled elements (icons, blank buttons) get visual indices.
Quote from commit: "Only show element IDs in highlights if llm_representation is less than 10 characters." (Code actually used < 5 chars.)
Bug found by cursor bot: The BrowserProfile description said "10 characters" but code used "5 characters" — a documentation mismatch.
Commit ce0876a — "highlight in parallel" (PR #2873)
What changed: Made create_highlighted_screenshot async and added @time_execution_async timing decorators everywhere — on the highlighting, DOM building, and CDP session creation. This was a performance investigation.
Motivation: The team was measuring how long highlighting took and trying to squeeze latency. The timing showed highlighting was a bottleneck, which motivated the later decision to remove it from the LLM path entirely.
Commit a31ae9d — "Refactor bounding box drawing and enhance LLM text representation"
What changed: Improved both the visual highlighting AND the text DOM representation simultaneously.
Motivation: The team was experimenting with whether better text representation could compensate for less aggressive visual annotation. This is the inflection point where they started investing more in the text DOM quality as an alternative to image annotation.
Agent Task Evaluation: 2/3 (67%)
The PR ran browser-use's automated benchmark (3 agent tasks). It passed 2 out of 3 — 67% success rate with annotated screenshots. This became the baseline they'd compare against when removing highlights later.

PR #2943: "fix-pil-import" (Sep 3-4, 2025)
This is the PR that removed highlights from the LLM path.

Commit e7897a9 — "Remove overlays"
What changed: Only 1 file, -24 lines. Removed the remove_screenshot_overlays() function — a CDP-based JavaScript function that cleaned up screenshot overlay divs from the page. This was dead code cleanup after the decision to stop annotating screenshots.
Key context: This commit was bundled with PIL fixes, suggesting the team discovered the highlighting code was causing practical problems (PIL import errors, memory leaks from font loading) AND wasn't improving agent performance enough to justify the complexity.
Evaluation: Same PR also scored 2/3 (67%) on the benchmark — identical to the annotated version. This is the critical data point: removing visual annotations from screenshots did not hurt agent success rate.
Commit 4f891cd — "Remove gc"
Removed forced gc.collect() calls that were added to combat memory leaks from PIL image processing for highlighting. The cubic-dev-ai bot had flagged this as causing "unnecessary pauses and CPU overhead."
Commit 53e5858 — "fix-pil-import"
Made Pillow a runtime dependency (moved from dev-only) and added cross-platform font loading with caching. This fixed crashes on systems without the expected fonts.
Commit 56ab048 — "Fix highlighting system with browser-side highlights" (Sep 18, 2025, PR #3151)
What changed: Added BrowserSession.add_highlights() — a new system that injects dashed-outline highlights into the actual browser DOM (not the screenshot image). These are visible to a human watching the browser, but the ScreenshotWatchdog explicitly calls remove_highlights() before capturing the screenshot for the LLM.
Architecture shift: Highlights are now browser-side DOM injection for human observers only. The LLM gets a clean screenshot.
How it works: After DOM build + clean screenshot are captured, the watchdog optionally calls add_highlights() to inject <div> overlays with dashed borders and index labels into the live page. Before the next screenshot, these are removed.
Screenshots sent to LLM: 1 clean screenshot (no annotations). DOM text provides indices.
Commits d319eef → 3aa5cfd — Rename to dom_highlight_elements (Sep 29, 2025)
What changed: The highlight_elements profile flag was renamed first to debug_highlight_elements, then to dom_highlight_elements.
Motivation: The naming evolution tells the story:
highlight_elements — original generic name (when highlights were for LLM)
debug_highlight_elements — explicit that it's for debugging (human viewing)
dom_highlight_elements — final name, clarifying it's DOM injection, not image annotation
Commit b678f27 — "Migration from highlight index to backendnode id" (Oct 22, 2025, PR #3377)
What changed: 12 files, massive refactor. Eliminated the element_index (sequential counter 1, 2, 3...) and replaced it with backend_node_id (Chrome's stable internal node identifier) throughout the entire codebase.
Key changes:
SimplifiedNode.interactive_index: int | None → SimplifiedNode.is_interactive: bool
EnhancedDOMTreeNode.element_index — field deleted entirely
Selector map key changed from sequential counter to backend_node_id
Serializer output: [{interactive_index}] → [{backend_node_id}]
All action handlers, pagination detection, dropdown handling — all switched from element_index to backend_node_id
Motivation: Sequential indices (1, 2, 3...) were fragile — any DOM change renumbered everything. backend_node_id is stable across CDP sessions for the same element. This also eliminates the small-number visual confusion problem entirely: backend_node_id values are large numbers like 742 that are unlikely to be confused with notification badges.
Screenshots sent to LLM: Still 1 clean screenshot. The text DOM now uses [742]<button> instead of [3]<button>.
Summary Table
Commit	Date	Screenshots to LLM	Highlight approach	Benchmark
6ad7e70 (PR #2705)	Aug 18	1 annotated (PIL bboxes + indices)	PIL on screenshot image	2/3 (67%)
e8bd858	Aug 30	1 annotated (filtered: only show index if text < 5 chars)	PIL, selective labeling	—
ce0876a	Aug 30	1 annotated (async parallelized)	PIL, performance-tuned	—
e7897a9 (PR #2943)	Sep 4	1 clean (no annotations)	None for LLM	2/3 (67%)
56ab048 (PR #3151)	Sep 18	1 clean	Browser DOM injection (human only)	—
b678f27 (PR #3377)	Oct 22	1 clean	DOM injection + backend_node_id	—
Key finding: The benchmark score was identical (67%) with and without screenshot annotations, which validated removing them. No published A/B results beyond this CI gate were found in the public PRs, but the team's behavior (progressive removal, rapid iteration) suggests internal testing confirmed no regression.
