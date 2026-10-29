"""
API Routes Package.
"""

from api.routes.email_routes import router as email_router
from api.routes.doc_routes import router as doc_router

__all__ = ["email_router", "doc_router"]
