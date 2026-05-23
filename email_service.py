"""Welcome email service for new user registrations.

Sends a branded HTML welcome email when a new user registers.
Requires SMTP_EMAIL and SMTP_PASSWORD in .env (Gmail app password recommended).
If SMTP is not configured, registration still succeeds — the email is simply skipped.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env if present
ENV_PATH = Path(__file__).resolve().parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=True)
except ImportError:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_smtp_config() -> tuple[str, str] | None:
    """Return (email, password) tuple or None if not configured."""
    email = os.environ.get("SMTP_EMAIL", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    if email and password and password not in ("your_app_password_here", ""):
        return email, password
    return None


def _build_welcome_html(full_name: str, email: str, password: str) -> str:
    """Build a professional HTML welcome email."""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f3f6fb;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="padding:32px 16px;">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(15,23,42,0.1);">

        <!-- Header -->
        <tr><td style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:32px 28px;text-align:center;">
          <h1 style="margin:0 0 6px;color:#ffffff;font-size:22px;">Welcome to Fraud Detection</h1>
          <p style="margin:0;color:rgba(226,232,240,0.8);font-size:14px;">Online Recruitment Fraud Detection System</p>
        </td></tr>

        <!-- Body -->
        <tr><td style="padding:28px;">
          <p style="margin:0 0 16px;font-size:15px;color:#0f172a;">
            Hi <strong>{full_name}</strong>,
          </p>
          <p style="margin:0 0 20px;font-size:15px;color:#334155;line-height:1.6;">
            Thank you for registering on our Online Recruitment Fraud Detection platform.
            Your account has been created successfully. Below are your account details:
          </p>

          <!-- Credentials Card -->
          <table width="100%" cellpadding="0" cellspacing="0" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;margin:0 0 20px;">
            <tr><td style="padding:18px 20px;">
              <p style="margin:0 0 10px;font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;">Your Account Details</p>
              <table cellpadding="0" cellspacing="0" style="width:100%;">
                <tr>
                  <td style="padding:6px 0;color:#64748b;font-size:14px;width:100px;">Name</td>
                  <td style="padding:6px 0;color:#0f172a;font-size:14px;font-weight:600;">{full_name}</td>
                </tr>
                <tr>
                  <td style="padding:6px 0;color:#64748b;font-size:14px;">Email</td>
                  <td style="padding:6px 0;color:#0f172a;font-size:14px;font-weight:600;">{email}</td>
                </tr>
                <tr>
                  <td style="padding:6px 0;color:#64748b;font-size:14px;">Password</td>
                  <td style="padding:6px 0;color:#0f172a;font-size:14px;font-weight:600;">{password}</td>
                </tr>
              </table>
            </td></tr>
          </table>

          <p style="margin:0 0 20px;font-size:14px;color:#64748b;line-height:1.6;">
            You can now log in and start using the fraud detection tools — upload CSV files,
            check individual job descriptions, and chat with our AI assistant.
          </p>

          <!-- CTA Button -->
          <table cellpadding="0" cellspacing="0" style="margin:0 auto 20px;">
            <tr><td style="background:linear-gradient(135deg,#2563eb,#4f8ef7);border-radius:12px;padding:12px 28px;">
              <a href="#" style="color:#ffffff;font-size:15px;font-weight:700;text-decoration:none;">Login to Your Account</a>
            </td></tr>
          </table>

          <p style="margin:0;font-size:13px;color:#94a3b8;line-height:1.5;text-align:center;">
            If you did not register for this account, please ignore this email.
          </p>
        </td></tr>

        <!-- Footer -->
        <tr><td style="background:#f8fafc;padding:18px 28px;border-top:1px solid #e2e8f0;text-align:center;">
          <p style="margin:0;font-size:12px;color:#94a3b8;">
            Online Recruitment Fraud Detection &mdash; B.Tech Final Year Project
          </p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_welcome_email(full_name: str, email: str, password: str) -> bool:
    """Send a welcome email to a newly registered user.

    Returns True if sent successfully, False otherwise.
    Silently skips if SMTP is not configured.
    """
    config = _get_smtp_config()
    if config is None:
        logger.info("SMTP not configured — skipping welcome email for %s", email)
        return False

    sender_email, sender_password = config

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Welcome to Online Recruitment Fraud Detection"
        msg["From"] = f"Fraud Detection System <{sender_email}>"
        msg["To"] = email

        html_body = _build_welcome_html(full_name, email, password)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        # Plain text fallback
        plain = (
            f"Welcome, {full_name}!\n\n"
            f"Your account has been created successfully.\n\n"
            f"Account Details:\n"
            f"  Name: {full_name}\n"
            f"  Email: {email}\n"
            f"  Password: {password}\n\n"
            f"You can now log in and start using the fraud detection tools.\n\n"
            f"— Online Recruitment Fraud Detection"
        )
        msg.attach(MIMEText(plain, "plain", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())

        logger.info("Welcome email sent to %s", email)
        return True

    except Exception as exc:
        logger.warning("Failed to send welcome email to %s: %s", email, exc)
        return False
