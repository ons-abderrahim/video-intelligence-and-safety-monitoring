"""
visp.services.alert
~~~~~~~~~~~~~~~~~~~
Dispatches alert notifications via webhook, email, and Slack
when a safety event is detected above the confidence threshold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from backend.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class AlertPayload:
    camera_id: str
    event_type: str
    confidence: float
    timestamp: str
    frame_id: int
    zone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def slack_text(self) -> str:
        severity = "🔴" if self.confidence >= 0.9 else "🟡"
        return (
            f"{severity} *VISP Safety Alert*\n"
            f"• Camera: `{self.camera_id}`\n"
            f"• Event: `{self.event_type}`\n"
            f"• Confidence: `{self.confidence:.0%}`\n"
            f"• Time: `{self.timestamp}`"
            + (f"\n• Zone: `{self.zone}`" if self.zone else "")
        )


class AlertService:
    """
    Sends alert notifications over configured channels.
    Failures are logged but never re-raised to avoid disrupting
    the inference pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def dispatch(self, payload: AlertPayload) -> None:
        """Fire-and-forget dispatch to all configured channels."""
        tasks = []

        if self._settings.alert_webhook_url:
            tasks.append(self._send_webhook(payload))

        if self._settings.slack_bot_token:
            tasks.append(self._send_slack(payload))

        if self._settings.alert_email_to:
            tasks.append(self._send_email(payload))

        for coro in tasks:
            try:
                await coro
            except Exception as exc:
                logger.warning("Alert dispatch failed: %s", exc)

    async def _send_webhook(self, payload: AlertPayload) -> None:
        url = str(self._settings.alert_webhook_url)
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(url, json=payload.to_dict())
            resp.raise_for_status()
        logger.debug("Webhook alert sent to %s", url)

    async def _send_slack(self, payload: AlertPayload) -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {self._settings.slack_bot_token}"},
                json={"channel": "#safety-alerts", "text": payload.slack_text()},
            )
            resp.raise_for_status()
        logger.debug("Slack alert sent")

    async def _send_email(self, payload: AlertPayload) -> None:
        try:
            import aiosmtplib
            from email.message import EmailMessage
        except ImportError:
            logger.warning("aiosmtplib not installed; skipping email alert")
            return

        msg = EmailMessage()
        msg["From"] = "visp-alerts@yourdomain.com"
        msg["To"] = self._settings.alert_email_to
        msg["Subject"] = f"[VISP] {payload.event_type} — {payload.camera_id}"
        msg.set_content(json.dumps(payload.to_dict(), indent=2))

        await aiosmtplib.send(
            msg,
            hostname=self._settings.smtp_host,
            port=self._settings.smtp_port,
            username=self._settings.smtp_user,
            password=self._settings.smtp_password,
        )
        logger.debug("Email alert sent to %s", self._settings.alert_email_to)
