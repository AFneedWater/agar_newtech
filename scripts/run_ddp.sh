#!/usr/bin/env bash
set -euo pipefail

DDP_LOG="${DDP_LOG:-1}"
DDP_LOG_STDOUT="${DDP_LOG_STDOUT:-0}"

_warn() {
  echo "[run_ddp] WARNING: $*" >&2
}

_script_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  (cd "${script_dir}/.." && pwd)
}

_init_log_dir() {
  local root ts
  root="$(_script_root)"
  ts="$(date +%Y%m%d_%H%M%S)"
  if [[ -z "${DDP_LOG_DIR:-}" ]]; then
    DDP_LOG_DIR="${root}/logs/ddp_run_${ts}"
  fi

  if ! mkdir -p "${DDP_LOG_DIR}" 2>/dev/null; then
    _warn "failed to create DDP_LOG_DIR=${DDP_LOG_DIR}; disable logging"
    DDP_LOG="0"
    return
  fi

  export DDP_LOG_DIR
  echo "[run_ddp] LOG_DIR=${DDP_LOG_DIR}"
}

_write_file() {
  # Usage: _write_file <path> <content>
  local path="$1"
  local content="$2"
  { printf "%s\n" "${content}" > "${path}"; } 2>/dev/null || _warn "failed to write ${path}"
}

_append_file() {
  # Usage: _append_file <path> <content>
  local path="$1"
  local content="$2"
  { printf "%s\n" "${content}" >> "${path}"; } 2>/dev/null || _warn "failed to write ${path}"
}

_join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf "%q " "${arg}")"
  done
  printf "%s" "${out% }"
}

if [[ "${DDP_LOG}" != "0" ]]; then
  _init_log_dir
fi

_is_bad_ifname() {
  local name="$1"
  [[ "${name}" == "lo" ]] && return 0
  [[ "${name}" == docker* ]] && return 0
  [[ "${name}" == br-* ]] && return 0
  [[ "${name}" == veth* ]] && return 0
  [[ "${name}" == tailscale* ]] && return 0
  [[ "${name}" == tun* ]] && return 0
  [[ "${name}" == wg* ]] && return 0
  [[ "${name}" == virbr* ]] && return 0
  [[ "${name}" == vmnet* ]] && return 0
  [[ "${name}" == ham* ]] && return 0
  [[ "${name}" == tap* ]] && return 0
  return 1
}

_pick_auto_ifname() {
  local name=""
  name="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
  if [[ -z "${name}" ]]; then
    name="$(ip route show default 2>/dev/null | awk '{print $5; exit}')"
  fi
  if [[ -n "${name}" ]] && _is_bad_ifname "${name}"; then
    name=""
  fi
  if [[ -z "${name}" ]]; then
    while read -r candidate; do
      if ip -o -4 addr show dev "${candidate}" >/dev/null 2>&1; then
        name="${candidate}"
        break
      fi
    done < <(ip -o link show up | awk -F': ' '{print $2}' | grep -E '^(enp|ens|eth|eno)' || true)
  fi
  echo "${name}"
}

IFNAME="${IFNAME:-}"
if [[ -z "${IFNAME}" ]]; then
  IFNAME="$(_pick_auto_ifname)"
  IFNAME_SOURCE="auto-detected"
else
  IFNAME_SOURCE="user-specified"
fi

if [[ -z "${IFNAME}" ]]; then
  echo "Failed to detect a suitable network interface" >&2
  exit 1
fi

echo "[run_ddp] IFNAME=${IFNAME} (${IFNAME_SOURCE})"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export IFNAME="${IFNAME}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFNAME}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFNAME}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DDP_PIN_CPU="${DDP_PIN_CPU:-0}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <command...>" >&2
  exit 2
fi

TOPO_RAW=""
TOPO_CLEAN=""
TOPO_LINES=()
TOPO_PARSED=""

if [[ "${DDP_LOG}" != "0" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    (nvidia-smi -L 2>/dev/null || true) > "${DDP_LOG_DIR}/nvidia-smi-L.txt" || _warn "failed to write nvidia-smi-L.txt"
    TOPO_RAW="$(nvidia-smi topo -m 2>/dev/null || true)"
    TOPO_CLEAN="$(printf '%s\n' "${TOPO_RAW}" | awk '{gsub(/\x1B\[[0-9;]*[mK]/, "", $0); print}')"
    _write_file "${DDP_LOG_DIR}/topo-m.txt" "${TOPO_CLEAN}"
  else
    _warn "nvidia-smi not found; skip nvidia-smi logs"
    _write_file "${DDP_LOG_DIR}/nvidia-smi-L.txt" "nvidia-smi not found"
    _write_file "${DDP_LOG_DIR}/topo-m.txt" "nvidia-smi not found"
  fi
fi

_extract_cpu_affinity_awk() {
  local topo_raw="$1"
  local gpu_idx="$2"
  local header
  local gpu_line
  local col

  header="$(printf '%s\n' "${topo_raw}" | awk '$1=="GPU0" && $2 ~ /^GPU/ {print; exit}')"
  if [[ -z "${header}" ]]; then
    header="$(printf '%s\n' "${topo_raw}" | awk '/CPU[[:space:]]+Affinity/ {print; exit}')"
  fi
  if [[ -z "${header}" ]]; then
    echo ""
    return
  fi

  col="$(printf '%s\n' "${header}" | awk '{for (i=1;i<=NF;i++) if ($i=="CPU" && $(i+1)=="Affinity") {print i+1; exit}}')"
  if [[ -z "${col}" ]]; then
    echo ""
    return
  fi

  gpu_line="$(printf '%s\n' "${topo_raw}" | awk -v g="GPU${gpu_idx}" '$1==g && $2 !~ /^GPU/ {print; exit}')"
  if [[ -z "${gpu_line}" ]]; then
    gpu_line="$(printf '%s\n' "${topo_raw}" | awk -v g="GPU${gpu_idx}" '$1==g {print; exit}')"
  fi
  if [[ -z "${gpu_line}" ]]; then
    echo ""
    return
  fi

  printf '%s\n' "${gpu_line}" | awk -v c="${col}" '{print $c}'
}

_extract_cpu_affinity() {
  local topo_raw="$1"
  local gpu_idx="$2"
  local parsed=""

  if command -v python >/dev/null 2>&1; then
    parsed="$(printf '%s\n' "${topo_raw}" | python -m agar.tools.parse_topo_affinity --gpu "${gpu_idx}" 2>/dev/null || true)"
  fi

  if [[ -z "${parsed}" ]]; then
    parsed="$(_extract_cpu_affinity_awk "${topo_raw}" "${gpu_idx}")"
  fi

  echo "${parsed}"
}

_get_topo_lines() {
  local topo_raw="$1"
  local header
  local gpu0
  local gpu1

  header="$(printf '%s\n' "${topo_raw}" | awk '$1=="GPU0" && $2 ~ /^GPU/ {print; exit}')"
  if [[ -z "${header}" ]]; then
    header="$(printf '%s\n' "${topo_raw}" | awk '/CPU[[:space:]]+Affinity/ {print; exit}')"
  fi
  gpu0="$(printf '%s\n' "${topo_raw}" | awk '$1=="GPU0" && $2 !~ /^GPU/ {print; exit}')"
  gpu1="$(printf '%s\n' "${topo_raw}" | awk '$1=="GPU1" && $2 !~ /^GPU/ {print; exit}')"

  printf '%s\n' "${header}" "${gpu0}" "${gpu1}"
}

_get_visible_gpu_list() {
  local list=()
  local raw="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -n "${raw}" ]]; then
    raw="${raw// /}"
    IFS=',' read -r -a list <<< "${raw}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while read -r idx; do
      list+=("${idx}")
    done < <(nvidia-smi -L 2>/dev/null | awk -F'[: ]+' '/^GPU [0-9]+:/ {print $2}')
  fi
  printf '%s\n' "${list[@]}"
}

if [[ "${DDP_LOG}" != "0" && -n "${TOPO_CLEAN}" ]]; then
  mapfile -t TOPO_LINES < <(_get_topo_lines "${TOPO_CLEAN}")
  if command -v python >/dev/null 2>&1; then
    TOPO_PARSED="$(printf '%s\n' "${TOPO_CLEAN}" | python -m agar.tools.parse_topo_affinity 2>/dev/null || true)"
  fi
fi

FINAL_CMD=("$@")

if [[ "${DDP_PIN_CPU}" == "1" ]]; then
  if [[ -z "${TOPO_CLEAN}" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      _warn "nvidia-smi not found; skip pinning"
    else
      TOPO_RAW="$(nvidia-smi topo -m 2>/dev/null || true)"
      TOPO_CLEAN="$(printf '%s\n' "${TOPO_RAW}" | awk '{gsub(/\x1B\[[0-9;]*[mK]/, "", $0); print}')"
    fi
  fi

  if [[ -n "${TOPO_CLEAN}" ]]; then
    mapfile -t TOPO_LINES < <(_get_topo_lines "${TOPO_CLEAN}")
    if [[ -n "${TOPO_LINES[0]:-}" ]]; then
      echo "[run_ddp] topo header: ${TOPO_LINES[0]}"
    fi
    if [[ -n "${TOPO_LINES[1]:-}" ]]; then
      echo "[run_ddp] topo GPU0: ${TOPO_LINES[1]}"
    fi
    if [[ -n "${TOPO_LINES[2]:-}" ]]; then
      echo "[run_ddp] topo GPU1: ${TOPO_LINES[2]}"
    fi
  fi

  if [[ "${FINAL_CMD[0]}" == "torchrun" ]]; then
    gpu_list=()
    while read -r idx; do
      [[ -n "${idx}" ]] && gpu_list+=("${idx}")
    done < <(_get_visible_gpu_list)

    if [[ "${#gpu_list[@]}" -eq 0 ]]; then
      _warn "failed to detect visible GPU list; skip pinning"
    else
      if command -v python >/dev/null 2>&1; then
        TOPO_PARSED="$(printf '%s\n' "${TOPO_CLEAN:-}" | python -m agar.tools.parse_topo_affinity 2>/dev/null || true)"
      fi

      declare -A affinity_map=()
      if [[ -n "${TOPO_PARSED}" ]]; then
        while IFS='=' read -r key val; do
          [[ -n "${key}" && -n "${val}" ]] && affinity_map["${key}"]="${val}"
        done <<< "${TOPO_PARSED}"
      fi

      affinities=()
      any_found=0
      msg="[run_ddp] CPU affinity:"
      for gpu_idx in "${gpu_list[@]}"; do
        affinity=""
        if [[ "${gpu_idx}" =~ ^[0-9]+$ ]]; then
          affinity="${affinity_map[GPU${gpu_idx}]:-}"
          if [[ -z "${affinity}" ]]; then
            affinity="$(_extract_cpu_affinity "${TOPO_CLEAN}" "${gpu_idx}")"
          fi
        fi
        if [[ -z "${affinity}" ]]; then
          affinities+=("")
          msg+=" GPU${gpu_idx}=<missing>"
          continue
        fi
        any_found=1
        affinities+=("${affinity}")
        msg+=" GPU${gpu_idx}=${affinity}"
      done

      if [[ "${any_found}" -eq 0 ]]; then
        _warn "failed to parse CPU affinity for all GPUs; skip pinning"
      else
        export DDP_CPU_AFFINITIES="$(IFS=';'; echo "${affinities[*]}")"
        export DDP_PIN_CPU="1"
        echo "${msg}"
        echo "[run_ddp] DDP_CPU_AFFINITIES=${DDP_CPU_AFFINITIES}"

        torchrun_args=()
        module_name=""
        module_args=()
        rest=("${FINAL_CMD[@]:1}")
        i=0
        while [[ "${i}" -lt "${#rest[@]}" ]]; do
          case "${rest[$i]}" in
            -m|--module)
              if [[ $((i + 1)) -ge "${#rest[@]}" ]]; then
                _warn "torchrun -m/--module requires a module name; skip pinning"
                module_name=""
                break
              fi
              module_name="${rest[$((i + 1))]}"
              module_args=("${rest[@]:$((i + 2))}")
              break
              ;;
            *)
              torchrun_args+=("${rest[$i]}")
              ;;
          esac
          i=$((i + 1))
        done
        if [[ -n "${module_name}" ]]; then
          FINAL_CMD=(torchrun "${torchrun_args[@]}" -m agar.tools.pin_and_exec -- -m "${module_name}" "${module_args[@]}")
        fi
      fi
    fi
  else
    _warn "DDP_PIN_CPU=1 but command is not torchrun; skip pinning"
  fi
fi

if [[ "${DDP_LOG}" != "0" ]]; then
  meta_path="${DDP_LOG_DIR}/meta.txt"
  env_path="${DDP_LOG_DIR}/env.txt"
  cmd_path="${DDP_LOG_DIR}/cmd.txt"
  topo_parsed_path="${DDP_LOG_DIR}/topo_parsed.txt"

  : > "${meta_path}" 2>/dev/null || _warn "failed to create ${meta_path}"
  _append_file "${meta_path}" "time=$(date)"
  _append_file "${meta_path}" "hostname=$(hostname)"
  _append_file "${meta_path}" "whoami=$(whoami)"
  _append_file "${meta_path}" "pwd=$(pwd)"
  _append_file "${meta_path}" "python=$(command -v python 2>/dev/null || true)"
  _append_file "${meta_path}" "torchrun=$(command -v torchrun 2>/dev/null || true)"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    _append_file "${meta_path}" "git_head=$(git rev-parse --short HEAD 2>/dev/null || true)"
    _append_file "${meta_path}" "git_status=$(git status --porcelain 2>/dev/null | tr '\n' ';' || true)"
  fi
  _append_file "${meta_path}" "IFNAME=${IFNAME} (${IFNAME_SOURCE})"
  _append_file "${meta_path}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  _write_file "${cmd_path}" "$(_join_cmd "${FINAL_CMD[@]}")"

  : > "${env_path}" 2>/dev/null || _warn "failed to create ${env_path}"
  for key in \
    CUDA_VISIBLE_DEVICES \
    IFNAME \
    GLOO_SOCKET_IFNAME \
    NCCL_SOCKET_IFNAME \
    NCCL_SOCKET_FAMILY \
    NCCL_NET \
    NCCL_IB_DISABLE \
    NCCL_SHM_DISABLE \
    NCCL_P2P_DISABLE \
    NCCL_CUMEM_HOST_ENABLE \
    NCCL_DEBUG \
    TORCH_DISTRIBUTED_DEBUG \
    OMP_NUM_THREADS \
    DDP_PIN_CPU \
    DDP_CPU_AFFINITIES \
    DDP_LOG \
    DDP_LOG_DIR \
    DDP_LOG_STDOUT; do
    _append_file "${env_path}" "${key}=${!key-}"
  done

  : > "${topo_parsed_path}" 2>/dev/null || _warn "failed to create ${topo_parsed_path}"
  if [[ -n "${TOPO_LINES[0]:-}" ]]; then
    _append_file "${topo_parsed_path}" "topo_header: ${TOPO_LINES[0]}"
  fi
  if [[ -n "${TOPO_LINES[1]:-}" ]]; then
    _append_file "${topo_parsed_path}" "topo_GPU0: ${TOPO_LINES[1]}"
  fi
  if [[ -n "${TOPO_LINES[2]:-}" ]]; then
    _append_file "${topo_parsed_path}" "topo_GPU1: ${TOPO_LINES[2]}"
  fi
  if [[ -n "${TOPO_PARSED:-}" ]]; then
    _append_file "${topo_parsed_path}" ""
    _append_file "${topo_parsed_path}" "parsed:"
    _append_file "${topo_parsed_path}" "${TOPO_PARSED}"
  fi
  _append_file "${topo_parsed_path}" ""
  _append_file "${topo_parsed_path}" "DDP_CPU_AFFINITIES=${DDP_CPU_AFFINITIES-}"
fi

if [[ "${DDP_LOG}" != "0" && "${DDP_LOG_STDOUT}" == "1" ]]; then
  set +e
  "${FINAL_CMD[@]}" 2>&1 | tee "${DDP_LOG_DIR}/torchrun.log"
  ec="${PIPESTATUS[0]}"
  set -e
  exit "${ec}"
fi

exec "${FINAL_CMD[@]}"
