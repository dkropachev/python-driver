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

"""
Internal driver event primitives.

This module intentionally does not expose a public subscription API.  It is a
small synchronous bus used to decouple driver subsystems that need to react to
shared internal state changes.
"""

from collections import defaultdict
import logging
from threading import RLock


log = logging.getLogger(__name__)


HOST = "HOST"

HOST_ADDED = "HOST_ADDED"
HOST_REMOVED = "HOST_REMOVED"
HOST_UP = "HOST_UP"
HOST_DOWN = "HOST_DOWN"
HOST_CHANGED = "HOST_CHANGED"


class DriverEvent(object):
    """
    Internal event envelope.
    """

    __slots__ = ("type", "category", "payload", "source")

    def __init__(self, event_type, category, payload=None, source=None):
        self.type = event_type
        self.category = category
        self.payload = payload
        self.source = source

    def __repr__(self):
        return "%s(type=%r, category=%r, payload=%r, source=%r)" % (
            self.__class__.__name__, self.type, self.category, self.payload, self.source)


class HostEventPayload(object):
    """
    Payload for host topology and runtime-state events.
    """

    __slots__ = (
        "host_id", "host", "old_host", "new_host", "changed_fields",
        "old_values", "new_values", "refresh_nodes")

    def __init__(self, host=None, host_id=None, old_host=None, new_host=None,
                 changed_fields=(), old_values=None, new_values=None,
                 refresh_nodes=True):
        if host is None:
            host = new_host if new_host is not None else old_host

        if host_id is None and host is not None:
            host_id = host.host_id

        self.host_id = host_id
        self.host = host
        self.old_host = old_host
        self.new_host = new_host
        self.changed_fields = tuple(changed_fields or ())
        self.old_values = old_values or {}
        self.new_values = new_values or {}
        self.refresh_nodes = refresh_nodes

    def __repr__(self):
        return ("%s(host_id=%r, host=%r, old_host=%r, new_host=%r, "
                "changed_fields=%r, old_values=%r, new_values=%r)") % (
                    self.__class__.__name__, self.host_id, self.host,
                    self.old_host, self.new_host, self.changed_fields,
                    self.old_values, self.new_values)


class _EventBus(object):
    """
    Synchronous internal event bus.
    """

    def __init__(self):
        self._type_subscribers = defaultdict(list)
        self._category_subscribers = defaultdict(list)
        self._lock = RLock()

    def subscribe(self, event_type, handler):
        with self._lock:
            handlers = self._type_subscribers[event_type]
            if handler not in handlers:
                handlers.append(handler)

    def unsubscribe(self, event_type, handler):
        with self._lock:
            self._remove_handler(self._type_subscribers.get(event_type), handler)

    def subscribe_category(self, category, handler):
        with self._lock:
            handlers = self._category_subscribers[category]
            if handler not in handlers:
                handlers.append(handler)

    def unsubscribe_category(self, category, handler):
        with self._lock:
            self._remove_handler(self._category_subscribers.get(category), handler)

    def publish(self, event):
        handlers = self._handlers_for_event(event)
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                log.exception("Error dispatching driver event %s to %r", event.type, handler)
        return event

    @staticmethod
    def _remove_handler(handlers, handler):
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def _handlers_for_event(self, event):
        with self._lock:
            raw_handlers = list(self._type_subscribers.get(event.type, ()))
            raw_handlers.extend(self._category_subscribers.get(event.category, ()))

        handlers = []
        for handler in raw_handlers:
            if handler not in handlers:
                handlers.append(handler)
        return handlers
