PY ?= python
B ?= 199
TOP_N ?= 300
DONORS ?= philosophy ponderings todayilearned

.PHONY: demo validate scope real

demo:
	$(PY) demo_synthetic.py --B $(B)

validate:
	$(PY) validate_synthetic.py --reps 10 --B 99

scope:
	$(PY) scope_report.py

# Scope definitions mirror writing/notes/decisions.md (recommendations, not yet decided).
ifeq ($(SCOPE),A_hackerclaw_general)
  START=2026-01-29
  END=2026-02-04
  T0=2026-01-31T11:00
endif
ifeq ($(SCOPE),B_zenity_general)
  START=2026-02-08
  END=2026-02-20
  T0=2026-02-12T00:00
endif

real:
	@test -n "$(START)" || (echo "set SCOPE=A_hackerclaw_general|B_zenity_general" && false)
	$(PY) -m pipeline.ingest_moltnet --start $(START) --end $(END) --submolts general \
	    --top-n $(TOP_N) --donor-submolts $(DONORS) --name $(SCOPE)
	$(PY) run_real.py --scope $(SCOPE) --t0 $(T0) --B $(B)

# Whole-platform run (Jan 31 - Feb 17, all submolts) with a planted ring.
PLATFORM=platform_jan31_feb17
platform-ingest:
	$(PY) -c "from pipeline.ingest_moltnet import build_platform, save; save(build_platform('2026-01-31', '2026-02-18', '2026-02-06T00:00'), '$(PLATFORM)')"
	$(PY) embed_scope.py --scope $(PLATFORM)

platform:
	$(PY) run_platform.py --scope $(PLATFORM) --m 12 --alpha 0.3 --B 99 --workers 24
	$(PY) run_platform.py --scope $(PLATFORM) --no-plant --B 99 --workers 24
