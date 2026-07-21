"""Organize — turn the catalog into action: tags, reorganization Plans, preview.

Phase 0 (tagging) + Phase 1 (preview-only reorganize) per
ORGANIZE_FEATURE_DESIGN.md. Nothing in this package writes to the filesystem:
tagging writes DB rows only, and the planner/preview are pure read + plan rows.
Transactional apply, the migration journal, and undo are Phase 2 and live
elsewhere when built.
"""
