#!/bin/bash
# S6: how bathos keys run provenance (read-only SQL against the local warm catalog).
set -u
echo "--- runs per project_slug for the aminx project and its consumer mpnn_ext"
bth sql "SELECT project_slug, count(*) AS n FROM runs WHERE project_slug IN ('aminx','mpnn_ext') GROUP BY 1 ORDER BY 1"
echo "--- columns of runs whose name mentions project"
bth sql "SELECT column_name FROM information_schema.columns WHERE table_name='runs' AND lower(column_name) LIKE '%project%'"
