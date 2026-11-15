from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Department(StrEnum):
    """The only allowed recipients. Each value is the department's email address."""

    HUMAN_RESOURCES = "human-resources@example.com"
    HELP_DESK = "help-desk@example.com"
    IT = "it@example.com"
    KADRY = "kadry@example.com"
    OTHER = "other@example.com"  # fallback for unrecognised requests


# What each department handles; rendered into the system prompt in this order (the more
# specific scope first: kadry before human-resources, help-desk before it).
DEPARTMENT_SCOPES: Final[Mapping[Department, str]] = MappingProxyType(
    {
        Department.KADRY: "administrative personnel matters: leave (urlop), "
        "sick leave (L4), employment contracts (umowa), payroll (wypłata), "
        "employment certificates (zaświadczenia)",
        Department.HUMAN_RESOURCES: "recruitment, training, development, benefits, "
        "team conflicts",
        Department.HELP_DESK: "first-line user support: broken computer, "
        "password reset, printer, email client",
        Department.IT: "infrastructure, servers, access and permission requests, "
        "buying or deploying software",
        Department.OTHER: "anything else (fallback)",
    }
)
