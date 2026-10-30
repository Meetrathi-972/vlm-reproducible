#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-scene_jsons}"
OUT_DIR="${2:-${ROOT}/SceneCritic_output}"
ASSET_DIR="${3:-}" 
#Pass mesh_dimension.json 

VERIFIER="SceneCritic_evaluator.py"
ONTOLOGY="ontology/ontology_output/ontology.json"
CONFIG_ROOT="ontology/ontology_output"
TASK_ROOT="benchmark/benchmark_base_rooms"

mkdir -p "$OUT_DIR"

RUN_LOG="${OUT_DIR}/run.log"
SUMMARY_TSV="${OUT_DIR}/summary_all.tsv"

# ── TSV header (overlap_obb_pct added after overlap_pct) ──
printf "group\tmethod\tscene_folder\troom\tid\tscene_json\tsp_score\tsp_weighted\tscale_pct\toverlap_pct\toverlap_obb_pct\torientation_pct\tcooccurrence_pct\tcompleteness_pct\tn_violations\tn_hard\tlog_file\tstatus\n" > "$SUMMARY_TSV"

# ── Run header ──
{
  echo "==== Semantic Verifier Eval (ALL scene_pos*.json) ===="
  echo "Date: $(date)"
  echo "ROOT: $ROOT"
  echo "OUT_DIR: $OUT_DIR"
  echo "ASSET_DIR: ${ASSET_DIR:-<not set>}"
  echo "VERIFIER: $VERIFIER"
  echo "ONTOLOGY: $ONTOLOGY"
  echo "CONFIG_ROOT: $CONFIG_ROOT"
  echo "TASK_ROOT: $TASK_ROOT"
  echo
} > "$RUN_LOG"

# ── Accumulators ──
sum_w=0; n=0
sum_w_p=0; np=0
sum_w_o=0; no=0

avg(){ awk -v s="$1" -v n="$2" 'BEGIN{ if(n==0) print "N/A"; else printf "%.4f", s/n }'; }
add(){ awk -v a="$1" -v b="$2" 'BEGIN{print a+b}'; }

# ── Infer room and id from folder name ──
infer_room_and_id() {
  local folder="$1"
  IFS="_" read -r -a parts <<< "$folder"
  local total="${#parts[@]}"
  local rid="${parts[-1]}"

  [[ "$rid" =~ ^[0-9]+$ ]] || { echo "NA NA"; return 0; }

  local half=$(( (total - 1) / 2 ))
  (( half >= 1 )) || half=1

  local room="${parts[0]}"
  for ((i=1;i<half;i++)); do
    room="${room}_${parts[i]}"
  done

  echo "$room $rid"
}

# ── Find all scene files, skip OUT_DIR ──
mapfile -t SCENES < <(
  find "$ROOT" -type f -name 'scene_pos*.json' \
    -not -path "${OUT_DIR}/*" \
  | sort
)

echo "Found ${#SCENES[@]} scene_pos*.json files." | tee -a "$RUN_LOG"

if (( ${#SCENES[@]} == 0 )); then
  echo "[ERROR] No scene_pos*.json found under ROOT=$ROOT" | tee -a "$RUN_LOG"
  echo "Try: find \"$ROOT\" -type f -name '*.json' | head -n 50" | tee -a "$RUN_LOG"
  exit 1
fi

total="${#SCENES[@]}"
processed=0

for scene in "${SCENES[@]}"; do
  ((++processed))

  scene_dir="$(dirname "$scene")"
  scene_folder="$(basename "$scene_dir")"

  # group/method extraction: ROOT/<group>/<method>/<scene_folder>/<scene.json>
  method_dir="$(dirname "$scene_dir")"
  method="$(basename "$method_dir")"
  group_dir="$(dirname "$method_dir")"
  group="$(basename "$group_dir")"

  read -r room rid <<< "$(infer_room_and_id "$scene_folder")"

  # ── Prepare output directory ──
  out_leaf="${OUT_DIR}/${group}/${method}/${scene_folder}/per_scene"
  mkdir -p "$out_leaf"
  logfile="${out_leaf}/$(basename "$scene" .json).log"

  if [[ "$room" == "NA" || "$rid" == "NA" ]]; then
    echo "[WARN] ($processed/$total) cannot infer room/id from folder='$scene_folder' (scene: $scene)" | tee -a "$RUN_LOG"
    printf "%s\t%s\t%s\tNA\tNA\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\tUNPARSEABLE\n" \
      "$group" "$method" "$scene_folder" "$scene" "$logfile" >> "$SUMMARY_TSV"
    continue
  fi

  config="${CONFIG_ROOT}/verifier_config_${room}.json"
  task="${TASK_ROOT}/${room}/${room}_${rid}.json"

  # ── Build command ──
  CMD=(python "$VERIFIER" "$scene"
       --config "$config"
       --task-config "$task"
       --ontology "$ONTOLOGY"
       --json)

  if [[ -n "$ASSET_DIR" ]]; then
    CMD+=(--asset-dir "$ASSET_DIR")
  fi

  # ── Print progress ──
  echo
  echo "[$processed/$total] >>> Running:"
  printf '%q ' "${CMD[@]}"
  echo
  echo "--------------------------------------------------"

  # ── Write log header ──
  {
    echo "PROGRESS: ${processed}/${total}"
    echo "GROUP: $group"
    echo "METHOD: $method"
    echo "SCENE_FOLDER: $scene_folder"
    echo "ROOM: $room"
    echo "ID: $rid"
    echo
    echo "COMMAND:"
    printf '%q ' "${CMD[@]}"
    echo -e "\n"
  } > "$logfile"

  # ── Validate config/task exist ──
  if [[ ! -f "$config" ]]; then
    echo "[WARN] ($processed/$total) missing config: $config" | tee -a "$RUN_LOG"
    echo "[WARN] missing config: $config" >> "$logfile"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\tMISSING_CONFIG\n" \
      "$group" "$method" "$scene_folder" "$room" "$rid" "$scene" "$logfile" >> "$SUMMARY_TSV"
    continue
  fi

  if [[ ! -f "$task" ]]; then
    echo "[WARN] ($processed/$total) missing task: $task" | tee -a "$RUN_LOG"
    echo "[WARN] missing task: $task" >> "$logfile"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\tMISSING_TASK\n" \
      "$group" "$method" "$scene_folder" "$room" "$rid" "$scene" "$logfile" >> "$SUMMARY_TSV"
    continue
  fi

  # ── Run verifier with --json output ──
  set +e
  json_output=$("${CMD[@]}" 2>>"$logfile")
  rc=$?
  set -e

  echo "$json_output" >> "$logfile"

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ($processed/$total) $group/$method/$scene_folder/$(basename "$scene") (rc=$rc)" | tee -a "$RUN_LOG"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\tFAIL\n" \
      "$group" "$method" "$scene_folder" "$room" "$rid" "$scene" "$logfile" >> "$SUMMARY_TSV"
    continue
  fi

  # ── Parse JSON output ──
  sp_score=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sp_score','NA'))" 2>/dev/null || echo "NA")
  sp_weighted=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sp_score_weighted','NA'))" 2>/dev/null || echo "NA")
  scale_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('scale','NA'))" 2>/dev/null || echo "NA")
  overlap_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('overlap','NA'))" 2>/dev/null || echo "NA")
  overlap_obb_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('overlap_obb','NA'))" 2>/dev/null || echo "NA")
  orientation_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('orientation','NA'))" 2>/dev/null || echo "NA")
  cooccurrence_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('cooccurrence','NA'))" 2>/dev/null || echo "NA")
  completeness_pct=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('by_rule',{}).get('completeness','NA'))" 2>/dev/null || echo "NA")
  n_violations=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('n_violations','NA'))" 2>/dev/null || echo "NA")
  n_hard=$(echo "$json_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('n_hard_violations','NA'))" 2>/dev/null || echo "NA")

  if [[ "$sp_weighted" == "NA" ]]; then
    echo "[PARSE FAIL] ($processed/$total) $group/$method/$scene_folder/$(basename "$scene")" | tee -a "$RUN_LOG"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t%s\tPARSE_FAIL\n" \
      "$group" "$method" "$scene_folder" "$room" "$rid" "$scene" "$logfile" >> "$SUMMARY_TSV"
    continue
  fi

  # ── Print + record (overlap_obb added to progress line) ──
  printf "OK  (%d/%d) %-10s %-35s %-28s  room=%-15s id=%-3s  W=%-7s ORI=%-7s OVL=%-7s OBB=%-7s CO=%-7s CMP=%-7s\n" \
    "$processed" "$total" "$group" "$method" "$scene_folder" "$room" "$rid" \
    "$sp_weighted" "$orientation_pct" "$overlap_pct" "$overlap_obb_pct" "$cooccurrence_pct" "$completeness_pct" \
    | tee -a "$RUN_LOG"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tOK\n" \
    "$group" "$method" "$scene_folder" "$room" "$rid" "$scene" \
    "$sp_score" "$sp_weighted" "$scale_pct" "$overlap_pct" "$overlap_obb_pct" "$orientation_pct" \
    "$cooccurrence_pct" "$completeness_pct" "$n_violations" "$n_hard" "$logfile" >> "$SUMMARY_TSV"

  # ── Accumulate weighted score ──
  sum_w="$(add "$sum_w" "$sp_weighted")"
  ((++n))

  if [[ "$room" == "bedroom" || "$room" == "living_room" ]]; then
    sum_w_p="$(add "$sum_w_p" "$sp_weighted")"
    ((++np))
  else
    sum_w_o="$(add "$sum_w_o" "$sp_weighted")"
    ((++no))
  fi
done

# ── Final summary ──
{
  echo
  echo "===== OVERALL ($n scenes) ====="
  echo "SP Weighted Avg: $(avg "$sum_w" "$n")"
  echo
  echo "===== PIVOTAL ($np) bedroom + living_room ====="
  echo "SP Weighted Avg: $(avg "$sum_w_p" "$np")"
  echo
  echo "===== OTHERS ($no) ====="
  echo "SP Weighted Avg: $(avg "$sum_w_o" "$no")"
  echo
  echo "Saved:"
  echo "  Summary TSV: $SUMMARY_TSV"
  echo "  Run log:     $RUN_LOG"
  echo "  Logs under:  $OUT_DIR/<group>/<method>/<scene_folder>/per_scene/"
} | tee -a "$RUN_LOG"

echo
echo "Done. Logs saved → $OUT_DIR"