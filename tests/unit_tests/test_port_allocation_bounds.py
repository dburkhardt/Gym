# SPDX-License-Identifier: Apache-2.0
import socket

import pytest

from nemo_gym.global_config import _find_open_port_using_range


@pytest.mark.parametrize("low,high,retries", [(0, 10, 2), (20, 10, 2), (1, 65536, 2), (1, 2, 0)])
def test_invalid_range_is_rejected(low, high, retries):
    with pytest.raises(ValueError):
        _find_open_port_using_range([], low, high, retries)


def test_exhausted_range_fails_without_sampling(monkeypatch):
    def forbidden(*args):
        raise AssertionError("An exhausted range must not sample or bind")

    monkeypatch.setattr("nemo_gym.global_config.sample", forbidden)
    with pytest.raises(RuntimeError, match="No candidate ports"):
        _find_open_port_using_range([10001, 10002], 10001, 10002)


def test_each_candidate_is_probed_at_most_once(monkeypatch):
    attempted = []

    class OccupiedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            attempted.append(address[1])
            raise OSError("occupied")

    monkeypatch.setattr("nemo_gym.global_config.socket", OccupiedSocket)
    with pytest.raises(RuntimeError, match="3 distinct"):
        _find_open_port_using_range([10002], 10001, 10004, 50)
    assert set(attempted) == {10001, 10003, 10004}
    assert len(attempted) == 3


def test_real_listener_refused_then_released_port_available():
    with socket.socket() as listener:
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="1 distinct"):
            _find_open_port_using_range([], port, port)
    assert _find_open_port_using_range([], port, port) == port
