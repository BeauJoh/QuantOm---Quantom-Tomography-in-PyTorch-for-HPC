#!/usr/bin/env bash

good() {
  echo -e "\e[0;32m$*\e[0m"
}

info() {
  echo -e "\e[0;35m$*\e[0m"
}

warning() {
  echo -e "\e[0;33m$*\e[0m"
}

error() {
  echo -e "\e[0;31m$*\e[0m" >&2
}

_have_boxes() {
  command -v boxes >/dev/null 2>&1
}

_print_banner_plain() {
  local msg="$1"
  local width=78
  local line
  printf -v line '%*s' "$width" ''
  line=${line// /-}

  printf "+%s+\n" "$line"
  printf "| %-${width}s |\n" "$msg"
  printf "+%s+\n" "$line"
}

_print_banner() {
  local msg="$1"

  if _have_boxes; then
    printf '%s\n' "$msg" | boxes -d shell
  else
    _print_banner_plain "$msg"
  fi
}

experiment() {
  _print_banner "$*"
}

section() {
  _print_banner "$*"
}

setup_runtime_paths() {
  local ROOT_DIR=$(pwd)
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_DIR/cpp:$ROOT_DIR/omp:$ROOT_DIR/sycl
}

export -f good info warning error experiment section

