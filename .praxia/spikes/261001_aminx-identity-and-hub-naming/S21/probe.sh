#!/bin/bash
# S21: how bathos keys run provenance (read-only SQL against the local warm catalog).
# Re-run of S20 with the claim stated without a run count that changes as runs are added.
set -u
echo "--- runs per project_slug for aminx, the hub working slug, and praxia"
bth sql "SELECT project_slug, count(*) AS n FROM runs WHERE project_slug IN ('aminx','aminx-hub','praxia') GROUP BY 1 ORDER BY 1"
echo "--- columns of runs whose name mentions project"
bth sql "SELECT column_name FROM information_schema.columns WHERE table_name='runs' AND lower(column_name) LIKE '%project%'"
