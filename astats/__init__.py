"""AStats — Agentic Statistical Analysis with Practitioner Feedback Loop"""
from .structure import infer_structure
from .assumptions import check_assumptions
from .tests import select_and_run
from .feedback import SessionManager
from .nl_parser import parse_query

__version__ = "0.1.0"
__all__ = ["infer_structure", "check_assumptions", "select_and_run", "SessionManager", "parse_query"]
