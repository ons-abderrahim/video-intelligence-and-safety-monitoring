"""
visp.api.routes.events
~~~~~~~~~~~~~~~~~~~~~~
REST endpoints for reading and managing safety events.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.dependencies import get_event_queue

router = APIRouter(prefix="/api/events", tags=["events"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class EventOut(BaseModel):
    id: str
    timestamp: datetime
    camera_id: str
    event_type: str
    confidence: float
    bounding_boxes: list[list[int]] = []
    labels: list[str] = []
    zone: str | None = None
    acknowledged: bool = False


class EventListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[EventOut]


class AcknowledgeRequest(BaseModel):
    acknowledged: bool = True


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=EventListResponse, summary="List safety events")
async def list_events(
    camera_id: str | None = Query(default=None, description="Filter by camera"),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    sort: Literal["asc", "desc"] = Query(default="desc"),
    event_queue=Depends(get_event_queue),
) -> EventListResponse:
    """
    Return paginated safety events with optional filters.
    Events are read from the Redis event queue.
    """
    events = await event_queue.list_events(
        camera_id=camera_id,
        event_type=event_type,
        min_confidence=min_confidence,
        sort=sort,
    )

    total = len(events)
    start = (page - 1) * page_size
    items = events[start : start + page_size]

    return EventListResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/{event_id}", response_model=EventOut, summary="Get a single event")
async def get_event(
    event_id: str,
    event_queue=Depends(get_event_queue),
) -> EventOut:
    event = await event_queue.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found")
    return event


@router.patch("/{event_id}/acknowledge", response_model=EventOut, summary="Acknowledge an event")
async def acknowledge_event(
    event_id: str,
    body: AcknowledgeRequest,
    event_queue=Depends(get_event_queue),
) -> EventOut:
    event = await event_queue.acknowledge_event(event_id, body.acknowledged)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found")
    return event


@router.delete("/{event_id}", status_code=204, summary="Delete an event")
async def delete_event(
    event_id: str,
    event_queue=Depends(get_event_queue),
) -> None:
    deleted = await event_queue.delete_event(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found")
