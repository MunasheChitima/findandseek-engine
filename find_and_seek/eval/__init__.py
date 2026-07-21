"""Search-quality evaluation harness.

Measures the hybrid search against a fixed gold set so every index/algorithm
change can be scored (hit@k, MRR, latency) instead of eyeballed. Run against a
*copy* of the index, never the live DB.
"""
