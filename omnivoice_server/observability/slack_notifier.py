"""
Slack notification for OmniVoice TTS.

Sends random sample notifications and quality alerts to Slack.
Non-blocking — all sends happen in a background thread.
"""

from __future__ import annotations

import logging
import os
import random
import socket
from concurrent.futures import ThreadPoolExecutor

import requests as http_requests

logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))
PROBABILITY = float(os.getenv("SLACK_NOTIFICATION_PROBABILITY", "0.1"))
HOSTNAME = socket.gethostname()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slack_notify")


def _should_send(has_quality_issues: bool) -> bool:
    if not WEBHOOK_URL:
        return False
    if has_quality_issues:
        return True
    return random.random() < PROBABILITY


def send_tts_notification(
    *,
    text: str,
    voice: str,
    mode: str,
    endpoint: str,
    trace_output: dict | None = None,
    trace_id: str | None = None,
) -> None:
    """Send TTS notification to Slack (non-blocking).

    Always sends if quality issues detected. Otherwise samples by probability.
    """
    quality = trace_output.get("quality", {}) if trace_output else {}
    assessment = quality.get("assessment", {}) if quality else {}
    performance = trace_output.get("performance", {}) if trace_output else {}
    signal = quality.get("signal", {}) if quality else {}
    health = trace_output.get("health", {}) if trace_output else {}

    has_issues = assessment.get("has_issues", False) or assessment.get("has_problems", False)

    if not _should_send(has_issues):
        return

    _executor.submit(
        _send_sync,
        text=text,
        voice=voice,
        mode=mode,
        endpoint=endpoint,
        performance=performance,
        quality=quality,
        signal=signal,
        assessment=assessment,
        health=health,
        trace_id=trace_id,
    )


def _send_sync(
    *,
    text: str,
    voice: str,
    mode: str,
    endpoint: str,
    performance: dict,
    quality: dict,
    signal: dict,
    assessment: dict,
    health: dict,
    trace_id: str | None,
) -> None:
    try:
        # Determine severity
        if assessment.get("has_problems"):
            emoji, severity = "🚨", "CRITICAL"
        elif assessment.get("has_issues"):
            emoji, severity = "⚠️", "WARNING"
        elif assessment.get("is_excellent"):
            emoji, severity = "🌟", "EXCELLENT"
        else:
            emoji, severity = "✅", "INFO"

        # Quality label
        quality_label = "Excellent" if assessment.get("is_excellent") else \
                        "Good" if assessment.get("is_good") else \
                        "Issues" if assessment.get("has_issues") else \
                        "Problems" if assessment.get("has_problems") else "OK"

        # Build metrics lines
        metrics_lines = []
        rtf = performance.get("rtf")
        if rtf is not None:
            metrics_lines.append(f"RTF: `{rtf:.3f}`")
        gen_ms = performance.get("generation_time_ms")
        dur_ms = performance.get("audio_duration_ms")
        if gen_ms and dur_ms:
            metrics_lines.append(f"Latency: `{gen_ms:.0f}ms` | Duration: `{dur_ms:.0f}ms`")

        # Signal metrics
        if signal:
            rms = signal.get("rms_db")
            clip = signal.get("clipping_percent", 0)
            dr = signal.get("dynamic_range_db")
            if rms is not None:
                metrics_lines.append(f"RMS: `{rms:.1f}dB` | DR: `{dr:.1f}dB` | Clip: `{clip:.2f}%`")

        # Duration outlier
        is_outlier = quality.get("is_outlier")
        if is_outlier:
            ratio = quality.get("duration_ratio", 0)
            metrics_lines.append(f"⚠️ Duration outlier: ratio=`{ratio:.2f}`")

        # Health
        success_rate = health.get("success_rate", 1.0)
        if success_rate < 1.0:
            metrics_lines.append(f"⚠️ Success rate: `{success_rate:.0%}`")

        # Text preview
        text_preview = text[:300] + ("..." if len(text) > 300 else "")

        # Langfuse link
        trace_url = f"{LANGFUSE_HOST}/trace/{trace_id}" if trace_id else None

        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} OmniVoice TTS", "emoji": True},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Endpoint:*\n`{endpoint}`"},
                    {"type": "mrkdwn", "text": f"*Voice:*\n`{voice}`"},
                    {"type": "mrkdwn", "text": f"*Mode:*\n`{mode}`"},
                    {"type": "mrkdwn", "text": f"*Quality:*\n{quality_label}"},
                    {"type": "mrkdwn", "text": f"*Host:*\n`{HOSTNAME}`"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                ],
            },
        ]

        if metrics_lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Metrics:*\n" + " | ".join(metrics_lines)},
            })

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Input:*\n```{text_preview}```"},
        })

        if trace_url:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{trace_url}|🔍 View Trace in Langfuse>"},
            })

        payload = {
            "text": f"{emoji} OmniVoice TTS — {severity} | {endpoint} | {voice}",
            "blocks": blocks,
        }

        resp = http_requests.post(WEBHOOK_URL, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Slack notification sent: %s %s", severity, endpoint)

    except Exception as e:
        logger.warning("Slack notification failed: %s", e)
