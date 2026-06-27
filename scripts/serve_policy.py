#!/usr/bin/env python
"""Thin CLI wrapper: ``python scripts/serve_policy.py --config pi0_aloha_sim``."""

from openpi_jax.policies.serve import main

if __name__ == "__main__":
    main()
