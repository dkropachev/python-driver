# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import Future
from threading import Event, Lock, Thread
from unittest.mock import Mock

import pytest

from cassandra.cluster import Cluster, ControlConnection, _LifecycleCoordinator
from cassandra.connection import DefaultEndPoint
from tests.unit.test_control_connection import (
    HOST_ID_1, HOST_ID_2, HOST_ID_3, MockConnection, _node_meta_results)


@pytest.fixture
def topology():
    cluster = Cluster()
    cluster.control_connection.shutdown()
    cluster._prepare_all_queries = Mock()
    cluster.profile_manager = Mock()
    cluster.profile_manager.distance.return_value = 0
    cluster._start_reconnector = Mock()
    for host_id, address in ((HOST_ID_1, '192.168.1.0'),
                             (HOST_ID_2, '192.168.1.1'),
                             (HOST_ID_3, '192.168.1.2')):
        host, _ = cluster.add_host(DefaultEndPoint(address), signal=False,
                                   host_id=host_id)
        host.set_up()
    connection = MockConnection()
    control = ControlConnection(cluster, 1, 2, 0, 0)
    control._connection = connection
    cluster.control_connection = control
    yield cluster, connection
    cluster.shutdown()


@pytest.mark.parametrize('event', ['up', 'down', 'add'])
def test_listener_refresh_during_concurrent_removal(topology, event):
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    entered = Event()
    published = Event()
    refreshed = Event()
    order = []
    errors = []
    listener = Mock()

    def callback(current):
        assert current is host
        entered.set()
        assert published.wait(5)
        assert current._is_removed
        cluster.refresh_nodes()
        refreshed.set()
        order.append(event)

    getattr(listener, 'on_' + event).side_effect = callback
    listener.on_remove.side_effect = lambda host: order.append('remove')
    cluster.register_listener(listener)

    def deliver():
        try:
            if event == 'up':
                result = Future()
                result.set_result(True)
                cluster._on_up_future_completed(host, {result}, [], Lock(), result)
            elif event == 'down':
                Cluster.on_down_potentially_blocking.__wrapped__(
                    cluster, host, False, host._down_event_generation)
            else:
                host._endpoint_add_retry_pending = True
                cluster.retry_failed_endpoint_add(host)
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=deliver)
    worker.start()
    try:
        assert entered.wait(5)
        del connection.peer_results[1][1]
        cluster.refresh_nodes()
        assert cluster.metadata.get_host_by_host_id(HOST_ID_3) is None
        assert host._is_removed
        assert order == []
    finally:
        published.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert refreshed.is_set()
    assert order == [event, 'remove']
    cluster._start_reconnector.assert_not_called()


def test_reentrant_removal_publishes_before_delivering_replacement(topology):
    cluster, connection = topology
    old = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    old_row = connection.peer_results[1].pop()
    order = []
    replacements = []

    def remove(host):
        assert host is old
        order.append('remove-start')
        connection.peer_results[1].append(old_row)
        cluster.refresh_nodes()
        replacement = cluster.metadata.get_host_by_host_id(HOST_ID_3)
        assert replacement is not old
        assert replacement._lifecycle_pending
        replacements.append(replacement)
        assert order == ['remove-start']
        order.append('remove-end')

    def add(host):
        assert host is replacements[0]
        order.append('add')

    cluster.profile_manager.on_remove.side_effect = remove
    cluster.profile_manager.on_add.side_effect = add
    cluster.refresh_nodes()
    assert order == ['remove-start', 'remove-end', 'add']
    assert old._is_removed
    assert not replacements[0]._lifecycle_pending
    assert replacements[0].is_up


def test_older_query_cannot_overwrite_newer_topology(topology):
    cluster, connection = topology
    queried = Event()
    release = Event()
    old_connection = MockConnection()
    del old_connection.peer_results[1][1]
    old_result = _node_meta_results(old_connection.local_results,
                                    old_connection.peer_results)
    errors = []

    def query(*args, **kwargs):
        queried.set()
        assert release.wait(5)
        return old_result

    old_connection.wait_for_responses.side_effect = query

    def refresh_old():
        try:
            cluster.control_connection._refresh_node_list_and_token_map(old_connection)
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=refresh_old)
    worker.start()
    try:
        assert queried.wait(5)
        current = cluster.metadata.get_host_by_host_id(HOST_ID_3)
        cluster.refresh_nodes()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert cluster.metadata.get_host_by_host_id(HOST_ID_3) is current
    assert not current._is_removed
    cluster.profile_manager.on_remove.assert_not_called()


def test_dispatcher_releases_ownership_after_callback_errors():
    coordinator = _LifecycleCoordinator()
    observed = []

    def fail():
        coordinator.submit(observed.append, 'nested')
        raise RuntimeError('listener failed')

    with coordinator.transaction():
        coordinator.submit(fail)
        coordinator.submit(observed.append, 'next')
    coordinator.submit(observed.append, 'last')
    assert observed == ['next', 'nested', 'last']


def test_publication_exception_still_delivers_committed_notifications():
    coordinator = _LifecycleCoordinator()
    observed = []
    with pytest.raises(ValueError):
        with coordinator.transaction():
            coordinator.submit(observed.append, 'committed')
            raise ValueError('later row rejected')
    assert observed == ['committed']


def test_notification_does_not_hold_coordination_mutex():
    coordinator = _LifecycleCoordinator()
    acquired = Event()

    def acquire():
        with coordinator.transaction():
            acquired.set()

    def callback():
        thread = Thread(target=acquire)
        thread.start()
        try:
            assert acquired.wait(5)
        finally:
            thread.join(5)
        assert not thread.is_alive()

    coordinator.submit(callback)
    assert acquired.is_set()


@pytest.mark.parametrize('with_future', [False, True])
def test_successful_up_releases_pending_addition_reservation(topology, with_future):
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    host.set_down()
    host._lifecycle_pending = True
    host._pending_host_addition = True
    if with_future:
        completed = Future()
        completed.set_result(True)
        cluster._on_up_future_completed(host, {completed}, [], Lock(), completed)
    else:
        cluster.on_up(host)
    assert host.is_up
    assert not host._lifecycle_pending
    assert not host._pending_host_addition


@pytest.mark.parametrize('policy_kind', ['dc', 'rack'])
def test_location_transition_removes_old_policy_membership(topology, policy_kind):
    from cassandra.policies import DCAwareRoundRobinPolicy, RackAwareRoundRobinPolicy
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    host.set_location_info('dc1', 'rack1')
    policy = (DCAwareRoundRobinPolicy('dc1', used_hosts_per_remote_dc=1)
              if policy_kind == 'dc' else
              RackAwareRoundRobinPolicy('dc1', 'rack1', used_hosts_per_remote_dc=1))
    policy.populate(cluster, [host])
    cluster.profile_manager.on_down.side_effect = policy.on_down
    cluster.profile_manager.on_up.side_effect = policy.on_up
    cluster.profile_manager.on_remove.side_effect = policy.on_remove
    with cluster._lifecycle.transaction():
        assert cluster.control_connection._update_location_info(host, 'dc2', 'rack2')
        assert host.datacenter == 'dc1'
    assert host.datacenter == 'dc2'
    assert host.rack == 'rack2'
    assert list(policy.make_query_plan()).count(host) == 1
    cluster.remove_host(host, refresh_nodes=False)
    assert host not in list(policy.make_query_plan())


@pytest.mark.parametrize('custom_callback', ['distance', 'reset'])
def test_custom_policy_refresh_does_not_hold_host_state_lock(topology, custom_callback):
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    entered = Event()
    published = Event()
    returned = Event()
    errors = []

    def callback(*args):
        entered.set()
        assert published.wait(5)
        cluster.refresh_nodes()
        returned.set()
        return 0

    if custom_callback == 'distance':
        cluster.profile_manager.distance.side_effect = callback
        del cluster._start_reconnector
        operation = lambda: Cluster.on_down_potentially_blocking.__wrapped__(
            cluster, host, False, host._down_event_generation)
    else:
        host.conviction_policy.reset = callback
        operation = lambda: cluster._finalize_add(host)

    def run():
        try:
            operation()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        del connection.peer_results[1][1]
        cluster.refresh_nodes()
        assert host._is_removed
    finally:
        published.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert returned.is_set()
    assert host._is_removed and not host.is_up
    assert host._reconnection_handler is None


def test_newer_location_reversal_supersedes_queued_change(topology):
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    host.set_location_info('dc1', 'rack1')
    control = cluster.control_connection
    with cluster._lifecycle.transaction():
        assert control._update_location_info(host, 'dc2', 'rack2')
        assert control._update_location_info(host, 'dc1', 'rack1')
    assert (host.datacenter, host.rack) == ('dc1', 'rack1')
    assert host._pending_location is None


def test_location_delivery_replaces_stale_replica_cache(topology):
    from cassandra.metadata import KeyspaceMetadata, Murmur3Token
    cluster, connection = topology
    metadata = cluster.metadata
    host = metadata.get_host_by_host_id(HOST_ID_3)
    host.set_location_info('dc1', 'rack1')
    metadata.keyspaces['ks'] = KeyspaceMetadata(
        'ks', True, 'NetworkTopologyStrategy', {'dc2': 1})
    with cluster._lifecycle.transaction():
        cluster.control_connection._update_location_info(host, 'dc2', 'rack2')
        metadata.rebuild_token_map('Murmur3Partitioner', {host: ['0']})
        old_map = metadata.token_map
        assert old_map.get_replicas('ks', Murmur3Token(0)) == []
    assert metadata.token_map is not old_map
    assert metadata.token_map.get_replicas('ks', Murmur3Token(0)) == [host]
    # Even late publication of old calculated caches cannot poison the new map.
    old_map.tokens_to_hosts_by_ks['ks'] = {Murmur3Token(0): []}
    assert metadata.token_map.get_replicas('ks', Murmur3Token(0)) == [host]


def test_repeated_location_destination_keeps_latest_reservation(topology):
    cluster, connection = topology
    host = cluster.metadata.get_host_by_host_id(HOST_ID_3)
    host.set_location_info('dc1', 'rack1')
    control = cluster.control_connection
    requested = []

    def on_up(current):
        if current.datacenter == 'dc3' and not requested:
            requested.append(True)
            assert control._update_location_info(host, 'dc3', 'rack1')

    cluster.profile_manager.on_up.side_effect = on_up
    with cluster._lifecycle.transaction():
        for dc in ('dc2', 'dc3', 'dc2'):
            assert control._update_location_info(host, dc, 'rack1')
    assert host.datacenter == 'dc3'
    assert host._pending_location is None
