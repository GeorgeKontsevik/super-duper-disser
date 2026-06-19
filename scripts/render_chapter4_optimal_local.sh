#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${ROOT_DIR}/itmo-phd-thesis-template-en"
BUILD_DIR="${ROOT_DIR}/outputs/chapter4-optimal-local/build"
OUT_DIR="${ROOT_DIR}/outputs/chapter4-optimal-local"
PDF_NAME="chapter4-optimal-local.pdf"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic is required. Install it with: brew install tectonic" >&2
  exit 1
fi

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}" "${OUT_DIR}"

rsync -a --delete \
  --exclude '.git' \
  --exclude '*.aux' \
  --exclude '*.bbl' \
  --exclude '*.bcf' \
  --exclude '*.blg' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.run.xml' \
  --exclude '*.toc' \
  --exclude 'thesis-itmo.pdf' \
  "${TEMPLATE_DIR}/" "${BUILD_DIR}/"

# Keep the source tree untouched; patch only the disposable build copy.
perl -0pi -e 's/\\setmainlanguage\[babelshorthands=true\]\{english\}/\\setmainlanguage{english}/' \
  "${BUILD_DIR}/common/fonts.tex"
perl -0pi -e 's/\\cyrdash/---/g' \
  "${BUILD_DIR}/common/setupsimple.tex" \
  "${BUILD_DIR}/common/styles.tex"
perl -0pi -e 's/\\renewcommand\*\{\\cftchaptername\}\{\\chaptername\\space\}/\\renewcommand*{\\cftchaptername}{Глава\\space}/' \
  "${BUILD_DIR}/common/styles.tex"
perl -0pi -e 's/\\renewcommand\*\{\\cftappendixname\}\{\\appendixname\\space\}/\\renewcommand*{\\cftappendixname}{Приложение\\space}/' \
  "${BUILD_DIR}/common/styles.tex"

(
  cd "${BUILD_DIR}"
  tectonic --keep-logs --keep-intermediates chapter4-optimal-local.tex
)

cp "${BUILD_DIR}/chapter4-optimal-local.pdf" "${OUT_DIR}/${PDF_NAME}"
echo "Wrote ${OUT_DIR}/${PDF_NAME}"
