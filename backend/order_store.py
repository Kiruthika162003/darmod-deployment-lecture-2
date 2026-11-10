"""Pending-order queue for the admin page.

When the agent calls submit_order, the before-tool callback queues a copy
in this store under the customer's session id. The admin page polls /admin
endpoints to list/approve/deny. Approvals flow back through the runner's
resume_after_approval path, which actually fires the real submit_order and
resumes the customer's chat.

In-memory only. Fine for a single-process lecture demo. Swap for Redis or
a real database in production.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

OrderStatus = Literal["pending", "approved", "denied"]


@dataclass
class Order:
    order_id: str
    session_id: str
    call_id: str
    items: list[str]
    name: str
    email: str
    total_cad: int
    queued_at: float
    status: OrderStatus = "pending"
    resolved_at: float | None = None
    reason: str | None = None
    # What Maple said to the customer after the decision. Filled in a few
    # seconds after approve/deny, once the resumed chat turn finishes.
    confirmation: str | None = None


class OrderStore:
    """Thread-safe in-memory order store."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._lock = RLock()

    def queue(
        self,
        session_id: str,
        call_id: str,
        items: list[str],
        name: str,
        email: str,
        total_cad: int,
    ) -> Order:
        order_id = uuid.uuid4().hex[:10]
        order = Order(
            order_id=order_id,
            session_id=session_id,
            call_id=call_id,
            items=list(items),
            name=name,
            email=email,
            total_cad=total_cad,
            queued_at=time.time(),
        )
        with self._lock:
            self._orders[order_id] = order
        return order

    def get(self, order_id: str) -> Order | None:
        with self._lock:
            return self._orders.get(order_id)

    def list(self, status: OrderStatus | None = None) -> list[Order]:
        with self._lock:
            xs = list(self._orders.values())
        if status is None:
            return sorted(xs, key=lambda o: o.queued_at, reverse=True)
        return [o for o in xs if o.status == status]

    def resolve(self, order_id: str, status: OrderStatus, reason: str | None = None) -> Order | None:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None
            order.status = status
            order.resolved_at = time.time()
            order.reason = reason
            return order

    def set_confirmation(self, order_id: str, text: str) -> None:
        with self._lock:
            order = self._orders.get(order_id)
            if order is not None:
                order.confirmation = text


_STORE = OrderStore()


def get_order_store() -> OrderStore:
    return _STORE
