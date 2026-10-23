`creator-evaluation.py.txt` is the unchanged evaluation-service controller from the
frozen molecular task s42256-024-00916-5 release bound by frozen_sources.json.
SHA256: 1746282f390dd54761a91b7b5836b962e9483dc7394149dafd69dd51ac247272.
It is retained as a source-patching/controller regression fixture, not a new
scientific scorer. Tests inject fixture-only scoring math and timer stubs; the
actual patched controller, input snapshot implementation and record paths execute.
