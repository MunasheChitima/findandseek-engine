"""Service supervision — run the API/worker/watcher as launchd agents.

Turns the three CLI entry points into login-time, auto-restarting background
services so the app is "installed" rather than hand-started. See `launchd.py`.
"""
