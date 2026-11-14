"""Send a user's message to a department over SMTP.

Named ``mailer`` rather than ``email`` so it can never shadow the stdlib package.
"""

from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import aiosmtplib
from email_validator import validate_email

from message_router.config import Settings
from message_router.departments import Department

SUBJECT_PREFIX = "[message-router]"


class EmailDeliveryError(Exception):
    """The SMTP server was unreachable, timed out, or refused the email."""

    def __init__(self, department: Department, reason: str) -> None:
        # Never include the body or the requester email: this text may end up in logs.
        super().__init__(f"Could not deliver email to {department.value}: {reason}")
        self.department = department
        self.reason = reason


def normalize_requester_email(value: str) -> str:
    """Return one ASCII addr-spec with a lowercased domain.

    Rejects display names, address lists, CR/LF, non-ASCII local parts (MailHog has
    no SMTPUTF8) and special-use domains. IDN domains become punycode. No DNS lookups.

    Raises ValueError (``EmailNotValidError`` is a subclass).
    """
    validated = validate_email(value, check_deliverability=False, allow_smtputf8=False)
    ascii_email = validated.ascii_email
    if ascii_email is None:  # always set with allow_smtputf8=False; narrows the type
        raise ValueError("The email address has no ASCII form.")
    # "=?...?=" is a legal local part, but EmailMessage decodes it as an RFC 2047
    # encoded word, which could turn it into other addresses or a display name.
    if "=?" in ascii_email:
        raise ValueError("Encoded words are not allowed in the email address.")
    return ascii_email


def build_message(
    department: Department, requester_email: str, message: str, *, sender: str
) -> EmailMessage:
    """Build the email to ``department`` with ``Reply-To`` set to the requester.

    Pure, no I/O. Headers come only from code; ``message`` becomes the body as is.

    Raises ValueError for an unknown department, an invalid requester email or an
    empty message.
    """
    department = Department(department)
    requester = normalize_requester_email(requester_email)
    if not message.strip():
        raise ValueError("The message must not be empty.")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = department.value
    msg["Reply-To"] = requester
    # Defence in depth: the header must hold exactly the validated address.
    reply_to = msg["Reply-To"].addresses
    if (
        len(reply_to) != 1
        or reply_to[0].display_name
        or (reply_to[0].addr_spec != requester)
    ):
        raise ValueError("The email address does not survive header encoding.")
    msg["Subject"] = f"{SUBJECT_PREFIX} Message from {requester}"
    msg["Date"] = formatdate(localtime=True)
    # An explicit domain: without it make_msgid() calls socket.getfqdn(), which can
    # block for seconds on a host with broken DNS.
    msg["Message-ID"] = make_msgid(domain=sender.rpartition("@")[2])
    msg.set_content(message)  # text/plain; charset=utf-8
    return msg


async def send_department_email(
    department: Department, requester_email: str, message: str, *, settings: Settings
) -> str:
    """Build and send the email; return its Message-ID.

    Raises ValueError for invalid input (before any I/O) and EmailDeliveryError for
    any SMTP failure. No retries: sending is not idempotent.
    """
    department = Department(department)
    msg = build_message(department, requester_email, message, sender=settings.smtp_from)
    try:
        # Explicit envelope: the requester (Reply-To) is never a recipient.
        await aiosmtplib.send(
            msg,
            sender=settings.smtp_from,
            recipients=[department.value],
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            timeout=settings.smtp_timeout,
        )
    except aiosmtplib.SMTPException as exc:
        raise EmailDeliveryError(department, f"{type(exc).__name__}: {exc}") from exc
    return str(msg["Message-ID"])
