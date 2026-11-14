from enum import StrEnum


class Department(StrEnum):
    """The only allowed recipients. Each value is the department's email address."""

    HUMAN_RESOURCES = "human-resources@example.com"
    HELP_DESK = "help-desk@example.com"
    IT = "it@example.com"
    KADRY = "kadry@example.com"
    OTHER = "other@example.com"  # fallback for unrecognised requests
