#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-showo2_drsa_cpu_preflight}"
MIN_HEADROOM_MB="${MIN_HEADROOM_MB:-128}"
INTERVAL="${INTERVAL:-30}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION"
    exit 0
fi

COMMAND="set -u; min=$((MIN_HEADROOM_MB * 1024 * 1024)); while :; do max=\$(cat /sys/fs/cgroup/memory.max); now=\$(cat /sys/fs/cgroup/memory.current); if [ \"\$max\" = max ]; then break; fi; head=\$((max-now)); printf '[%s] waiting: headroom=%s MiB (need %s MiB)\\n' \"\$(date -u +%H:%M:%S)\" \"\$((head/1024/1024))\" '$MIN_HEADROOM_MB'; if [ \"\$head\" -ge \"\$min\" ]; then break; fi; sleep '$INTERVAL'; done; exec '$ROOT/evaluation/c4_drsa_adapter/run_cpu_preflight.sh'"

tmux new-session -d -s "$SESSION" -c "$ROOT" bash -lc "$COMMAND"
echo "Started or waiting in tmux: $SESSION"
echo "View: tmux capture-pane -t '=$SESSION' -p -S -80"
