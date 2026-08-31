"""Dashboard.

Reads a recorded run from disk. The simulator never imports this package, so
rendering cannot throttle simulation, and live viewing and replay are the same
code path -- one tails a run in progress, the other reads a finished one.
"""
