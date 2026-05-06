from concurrent.futures import Future
import gc
import logging
from threading import RLock
import time
import uuid
import weakref

from unittest.mock import ANY, Mock, patch

import pytest

from cassandra import ConsistencyLevel
from cassandra.cluster import (Cluster, ResponseFuture, Session,
                               _SessionHostEventHandler)
from cassandra.connection import DefaultEndPoint
from cassandra.events import (_EventBus, DriverEvent, HOST, HOST_ADDED,
                              HOST_CHANGED, HOST_DOWN, HostEventPayload)
from cassandra.metadata import Metadata
from cassandra.policies import (HostDistance, LoadBalancingPolicy,
                                RoundRobinPolicy, SimpleConvictionPolicy)
from cassandra.pool import Host
from cassandra.protocol import ProtocolHandler, QueryMessage
from cassandra.query import SimpleStatement


def _completed_future(result=True):
    future = Future()
    future.set_result(result)
    return future


def _wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


class _RecordingPolicy(LoadBalancingPolicy):

    def __init__(self):
        self.events = []
        self.hosts = []

    def distance(self, host):
        return HostDistance.LOCAL

    def populate(self, cluster, hosts):
        self.hosts = list(hosts)

    def make_query_plan(self, working_keyspace=None, query=None):
        return list(self.hosts)

    def on_up(self, host):
        self.events.append(("up", host))

    def on_down(self, host):
        self.events.append(("down", host))

    def on_add(self, host):
        self.events.append(("add", host))

    def on_remove(self, host):
        self.events.append(("remove", host))

    def on_change(self, old_host, new_host, changed_fields):
        self.events.append(("change", old_host, new_host, changed_fields))


class _FakeSession(object):

    def __init__(self):
        self.pools = {}
        self.update_created_pools_calls = 0

    def add_or_renew_pool(self, host, is_host_addition):
        self.pools[host.host_id] = host
        return _completed_future(True)

    def update_created_pools(self):
        self.update_created_pools_calls += 1

    def remove_pool(self, host):
        self.pools.pop(host.host_id, None)
        return _completed_future(True)

    def shutdown(self):
        pass


def test_event_bus_dispatches_type_then_category_and_dedupes():
    bus = _EventBus()
    calls = []

    def first(event):
        calls.append(("first", event.type))

    def second(event):
        calls.append(("second", event.category))

    bus.subscribe(HOST_ADDED, first)
    bus.subscribe_category(HOST, second)
    bus.subscribe_category(HOST, first)

    event = DriverEvent(HOST_ADDED, HOST, payload={"host": "h"}, source="test")
    assert bus.publish(event) is event
    assert calls == [("first", HOST_ADDED), ("second", HOST)]


def test_event_bus_unsubscribe_methods_are_idempotent():
    bus = _EventBus()
    calls = []

    def handler(event):
        calls.append(event.type)

    bus.subscribe(HOST_ADDED, handler)
    bus.subscribe_category(HOST, handler)
    bus.unsubscribe(HOST_ADDED, handler)
    bus.unsubscribe(HOST_ADDED, handler)
    bus.unsubscribe_category(HOST, handler)
    bus.unsubscribe_category(HOST, handler)

    bus.publish(DriverEvent(HOST_ADDED, HOST))
    assert calls == []


def test_event_bus_isolates_subscriber_exceptions(caplog):
    bus = _EventBus()
    calls = []

    def broken(event):
        raise RuntimeError("boom")

    def working(event):
        calls.append(event.type)

    bus.subscribe(HOST_ADDED, broken)
    bus.subscribe(HOST_ADDED, working)

    with caplog.at_level(logging.ERROR):
        bus.publish(DriverEvent(HOST_ADDED, HOST))

    assert calls == [HOST_ADDED]
    assert "Error dispatching driver event" in caplog.text


def test_host_identity_is_host_id_and_topology_fields_are_read_only():
    host_id = uuid.uuid4()
    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=host_id)
    same_identity = Host("127.0.0.2", SimpleConvictionPolicy, host_id=host_id)
    other_identity = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())

    assert host == same_identity
    assert hash(host) == hash(same_identity)
    assert host != other_identity

    with pytest.raises(AttributeError):
        host.endpoint = DefaultEndPoint("127.0.0.9")

    with pytest.raises(AttributeError):
        host.host_id = uuid.uuid4()

    with pytest.raises(AttributeError):
        host._datacenter = "dc2"


def test_set_location_info_returns_replacement_without_mutating_host():
    host = Host("127.0.0.1", SimpleConvictionPolicy, datacenter="dc1",
                rack="rack1", host_id=uuid.uuid4())

    replacement = host.set_location_info("dc2", "rack2")

    assert replacement is not host
    assert replacement.host_id == host.host_id
    assert replacement.runtime_state is host.runtime_state
    assert host.datacenter == "dc1"
    assert host.rack == "rack1"
    assert replacement.datacenter == "dc2"
    assert replacement.rack == "rack2"


def test_host_replacement_reuses_runtime_state_and_updates_endpoint_index():
    bus = _EventBus()
    events = []
    bus.subscribe(HOST_CHANGED, events.append)
    metadata = Metadata(bus)
    host_id = uuid.uuid4()
    host, _ = metadata.add_or_return_host(
        Host("127.0.0.1", SimpleConvictionPolicy, host_id=host_id))

    host.set_down()
    reconnector = object()
    sharding_info = object()
    host.get_and_set_reconnection_handler(reconnector)
    host.sharding_info = sharding_info
    new_host, changed_fields = metadata.replace_host(
        host_id, endpoint=DefaultEndPoint("127.0.0.2"), datacenter="dc1")

    assert changed_fields == ("endpoint", "datacenter")
    assert new_host is not host
    assert new_host == host
    assert new_host.runtime_state is host.runtime_state
    assert new_host.is_up is False
    assert new_host.sharding_info is sharding_info
    assert new_host.get_and_set_reconnection_handler(None) is reconnector
    assert metadata.get_host(DefaultEndPoint("127.0.0.1")) is None
    assert metadata.get_host(DefaultEndPoint("127.0.0.2")) is new_host
    assert events[-1].payload.old_host is host
    assert events[-1].payload.new_host is new_host


def test_sharding_info_change_is_runtime_host_changed_event():
    bus = _EventBus()
    events = []
    bus.subscribe(HOST_CHANGED, events.append)
    metadata = Metadata(bus)
    host_id = uuid.uuid4()
    host, _ = metadata.add_or_return_host(
        Host("127.0.0.1", SimpleConvictionPolicy, host_id=host_id))

    sharding_info = object()
    host.sharding_info = sharding_info

    assert host.sharding_info is sharding_info
    assert len(events) == 1
    assert events[0].payload.old_host is host
    assert events[0].payload.new_host is host
    assert events[0].payload.changed_fields == ("sharding_info",)
    assert events[0].payload.new_values["sharding_info"] is sharding_info


def test_public_on_add_fires_after_host_is_up_and_pools_are_ready():
    cluster = Cluster(protocol_version=4)
    cluster._prepare_all_queries = Mock()
    session = _FakeSession()
    cluster.sessions.add(session)
    observed = []

    class Listener(object):

        def on_add(self, host):
            observed.append((host.is_up, host.host_id in session.pools,
                             session.update_created_pools_calls))

        def on_up(self, host):
            pass

        def on_down(self, host):
            pass

        def on_remove(self, host):
            pass

    try:
        cluster.register_listener(Listener())
        host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4(),
                    event_bus=cluster._event_bus)
        cluster.metadata.add_or_return_host(host)

        cluster.on_add(host, refresh_nodes=False)

        assert observed == [(True, True, 1)]
    finally:
        cluster.shutdown()


def test_public_host_state_listener_fires_once_per_transition():
    cluster = Cluster(protocol_version=4)
    cluster._start_reconnector = Mock()
    observed = []

    class Listener(object):

        def on_add(self, host):
            observed.append("add")

        def on_up(self, host):
            observed.append("up")

        def on_down(self, host):
            observed.append("down")

        def on_remove(self, host):
            observed.append("remove")

    try:
        cluster.register_listener(Listener())
        host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4(),
                    event_bus=cluster._event_bus)
        cluster.metadata.add_or_return_host(host)

        cluster.on_add(host, refresh_nodes=False)
        host.set_down()
        cluster.on_up(host)
        cluster.on_down(host, is_host_addition=False)
        _wait_for(lambda: "down" in observed)
        cluster.on_remove(host)

        assert observed.count("add") == 1
        assert observed.count("up") == 1
        assert observed.count("down") == 1
        assert observed.count("remove") == 1
    finally:
        cluster.shutdown()


def test_lbp_receives_one_notification_per_host_transition_and_change():
    policy = _RecordingPolicy()
    cluster = Cluster(load_balancing_policy=policy, protocol_version=4)
    cluster._start_reconnector = Mock()

    try:
        host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4(),
                    event_bus=cluster._event_bus)
        cluster.metadata.add_or_return_host(host)

        cluster.on_add(host, refresh_nodes=False)
        host.set_down()
        cluster.on_up(host)
        cluster.on_down(host, is_host_addition=False)
        _wait_for(lambda: [event[0] for event in policy.events].count("down") == 1)
        cluster.on_remove(host)

        host.sharding_info = object()
        cluster.metadata.add_or_return_host(host)
        new_host, _ = cluster.metadata.replace_host(host.host_id, datacenter="dc2")

        event_names = [event[0] for event in policy.events]
        assert event_names.count("add") == 1
        assert event_names.count("up") == 1
        assert event_names.count("down") == 1
        assert event_names.count("remove") == 1
        assert event_names.count("change") == 1
        assert policy.events[-1] == ("change", host, new_host, ("datacenter",))
    finally:
        cluster.shutdown()


def test_topology_host_changed_replaces_round_robin_cached_host():
    policy = RoundRobinPolicy()
    cluster = Cluster(load_balancing_policy=policy, protocol_version=4)

    try:
        host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4(),
                    event_bus=cluster._event_bus)
        host.set_up()
        cluster.metadata.add_or_return_host(host)
        policy.populate(cluster, [host])

        new_host, _ = cluster.metadata.replace_host(host.host_id, datacenter="dc2")
        query_plan = list(policy.make_query_plan())

        assert any(candidate is new_host for candidate in query_plan)
        assert not any(candidate is host for candidate in query_plan)
    finally:
        cluster.shutdown()


def _session_for_pool_tests():
    session = Session.__new__(Session)
    session.cluster = Mock()
    session.cluster.connect_timeout = 1
    session.cluster.signal_connection_failure = Mock()
    session._profile_manager = Mock()
    session._profile_manager.distance.return_value = HostDistance.LOCAL
    session._pools = {}
    session._lock = RLock()
    session.keyspace = None
    session.submit = lambda fn, *args, **kwargs: _completed_future(fn(*args, **kwargs))
    return session


def test_session_pools_are_keyed_by_host_id():
    session = _session_for_pool_tests()
    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())

    class FakePool(object):

        def __init__(self, host, host_distance, session):
            self.host = host
            self.host_distance = host_distance
            self._keyspace = session.keyspace
            self.is_shutdown = False
            self.shutdown = Mock()

    with patch("cassandra.cluster.HostConnection", FakePool):
        assert session.add_or_renew_pool(host, is_host_addition=False).result() is True

    assert set(session._pools) == {host.host_id}
    assert session._pools[host.host_id].host is host


def test_session_rebinds_pool_for_non_endpoint_host_replacement():
    session = _session_for_pool_tests()
    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())
    new_host = host.copy_with(datacenter="dc2")
    pool = Mock()
    pool.is_shutdown = False
    pool.host_distance = HostDistance.LOCAL
    session._pools[host.host_id] = pool
    session.cluster.metadata.all_hosts.return_value = [new_host]

    session.on_change(host, new_host, ("datacenter",))

    pool.rebind_host.assert_called_once_with(new_host)


def test_session_renews_pool_for_endpoint_host_replacement():
    session = _session_for_pool_tests()
    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())
    new_host = host.copy_with(endpoint=DefaultEndPoint("127.0.0.2"))
    session._pools[host.host_id] = Mock()
    session.add_or_renew_pool = Mock(return_value="future")

    assert session.on_change(host, new_host, ("endpoint",)) == "future"
    session.add_or_renew_pool.assert_called_once_with(new_host, is_host_addition=False)


def test_response_future_pool_lookup_uses_host_id():
    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())
    session = Mock(spec=Session)
    session.keyspace = None
    session.row_factory = lambda column_names, rows: rows
    session.cluster.control_connection._tablets_routing_v1 = False
    session.cluster._default_load_balancing_policy.make_query_plan.return_value = [host]
    pool = Mock()
    session._pools.get.side_effect = {host.host_id: pool}.get
    connection = Mock()
    pool.is_shutdown = False
    pool.borrow_connection.return_value = (connection, 1)

    query = SimpleStatement("SELECT * FROM system.local")
    message = QueryMessage(query=query.query_string, consistency_level=ConsistencyLevel.ONE)
    future = ResponseFuture(session, message, query, 1)

    assert future.send_request() is True
    session._pools.get.assert_any_call(host.host_id)
    connection.send_msg.assert_called_once_with(
        future.message, 1, cb=ANY,
        encoder=ProtocolHandler.encode_message,
        decoder=ProtocolHandler.decode_message,
        result_metadata=[])


def test_session_host_event_handler_unsubscribes_after_session_gc():
    bus = _EventBus()

    class DummySession(object):

        def _handle_host_event(self, event):
            pass

    session = DummySession()
    handler = _SessionHostEventHandler(session, bus)
    session_ref = weakref.ref(session)

    del session
    gc.collect()

    assert session_ref() is None
    assert handler not in bus._type_subscribers[HOST_DOWN]

    host = Host("127.0.0.1", SimpleConvictionPolicy, host_id=uuid.uuid4())
    bus.publish(DriverEvent(HOST_DOWN, HOST, HostEventPayload(host=host)))
    assert handler not in bus._type_subscribers[HOST_DOWN]
