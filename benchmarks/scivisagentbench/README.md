# SciVisAgentBench: native Napari integration (in progress)

Creator revision `b9435d13e9ff979226b1754e6c467efeb026978a`. The first calibration is the actual visual case 1: both complete dataset002 TIFFs, red/green additive overlay, 3D and time14, screenshot plus answer. The native eager loader is retained. Full-resolution software rendering passed with peak RSS7,359,632KiB; this is a runtime control, not a model evaluation.

`native.py` invokes the creator's `NapariMCPAgent`, including its clearing request, efficiency instructions and TinyAgent loop. `transport.py` adapts the gateway's nonstreaming response to one native streaming chunk, preserving tool calls and usage. It rejects wrong-model, incomplete and empty replies after saving raw receipts, with no retries. Every LLM role must be `nvidia/nvidia/nemotron-3.5-super-vl`.

`responses_api_agents/scivis_native/app.py` is a serial Gym entry point. Its inference gate defaults closed, and results remain masked until the native evaluator has been wired and validated. **Do not enable or launch yet.** The server is not a scientifically validated integration.

The native MCP1.9.4 has two runtime incompatibilities: current Pydantic removes its private helper, and MCP calls `issubclass` on postponed string annotations in the creator's code. A separate RAM overlay pins Pydantic2.10.6/settings2.8.1. `mcp_bootstrap.py` resolves the existing annotations with `typing.get_type_hints` before native FastMCP registration. No native tool implementation or schema is intentionally changed. An attempted MCP1.12.4 upgrade had the same annotation bug and is not the selected runtime.

Actual MCP stdio/socket controls passed. The same restricted container passed full two-TIFF native eager loading, red/green additive overlay,3D/time14, screenshot, writable save_text, and denied GS/rubric/key/socket paths; peak RSS7,392,036KiB under16GiB/2CPU. Native evaluator text/vision parser and transport12 mocked fixtures pass, without scientific calibration.

Remaining gates: filesystem MCP and complete persistent viewer lifecycle; evaluator-only GS and rubric denial through every solver tool; native vision/text evaluator valid/wrong/empty/malformed controls; complete native Gym configuration/startup and mocked end-to-end run; independent review; provider quota clearance. The current worker root disk is near its8GiB reserve. Dependencies and exact TIFFs are in a bounded12GiB tmpfs, not copied into durable layers. No new machines or terms acceptance.
