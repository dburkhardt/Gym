# SPDX-License-Identifier: Apache-2.0
from benchmarks.bioagentbench.qualify_native_cli import MODEL, fixture_response, stream_chunks


def test_unique_fixture_response_identity_survives_stream_final():
    responses = [fixture_response({"model": MODEL}, i)[1] for i in range(3)]
    assert len({response["id"] for response in responses}) == 3
    for response in responses:
        chunks = stream_chunks(response)
        assert {chunk["id"] for chunk in chunks} == {response["id"]}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[0]["choices"][0]["delta"] == response["choices"][0]["message"]
