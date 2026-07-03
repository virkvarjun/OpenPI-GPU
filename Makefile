# Sharder convenience targets.
PY ?= python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)/packages/openpi-client/src

.PHONY: demo demo-naive demo-contrast figures test

# Interactive fault-tolerance demo (elastic supervisor). Kill a process and watch it resume exactly.
demo:
	$(PY) scripts/demo_fault_tolerance.py

# Interactive, WITHOUT the supervisor: a kill ends the run (the problem the supervisor solves).
demo-naive:
	$(PY) scripts/demo_fault_tolerance.py --mode naive

# Scripted, non-interactive: same kill in both modes back-to-back (naive dies, sharder resumes). Good for recording.
demo-contrast:
	$(PY) scripts/demo_fault_tolerance.py --contrast --script kill@step=6 --steps 14 --save-interval 3

# Regenerate the site result figures from the measured numbers.
figures:
	$(PY) scripts/plot_results.py

# Cheap-ladder tests (CPU / localhost multi-process).
test:
	$(PY) -m pytest src/openpi/training/data_sharding_test.py scripts/elastic_launch_test.py -q
