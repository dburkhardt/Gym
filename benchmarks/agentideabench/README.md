# AgentIdeaBench release validation

This is a maintained release-data integrity check, not a runnable Gym benchmark. The creator release at `39a310db1a21c83f71455565d05f485933f648bd` contains 51 tables, including results and derived evidence identifiers. It deliberately omits corpus text required for evaluation. Table rows must not be reported as task counts.

From the Gym root:

```sh
python -m benchmarks.agentideabench.release /path/to/release_data --manifest-sha256 c480d4fe097dd7cca1cb28259aaf051dfa877434261725d682373bdf830e4ec1
```

This checks the pinned creator manifest, each compressed file hash and size, exact CSV headers and row counts. It refuses traversal and symlinks and performs no acquisition or inference. A successful receipt preserves the creator redaction disclosure and explicitly says the inputs remain incomplete. The supplied checksum must come from an independently trusted release; computing it from untrusted inputs is not provenance verification.

Remaining work is a complete, authorized corpus with reference/evidence text, exact Static versus Active task selection, native solver and critic transport, and scientific scorer qualification. Reacquired literature is a retrieval-time variant. Three Nemotron critic samples must retain distinct slot identities; the creator database keys by critic model, so simply repeating the same model name would collide. No default dataset or new LLM protocol is invented here.
