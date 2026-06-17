#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${ROOT_DIR}/itmo-phd-thesis-template-en"
OVERLAY_DIR="${ROOT_DIR}/thesis"
BUILD_DIR="${ROOT_DIR}/outputs/thesis/build"
OUT_DIR="${ROOT_DIR}/outputs/thesis"
PDF_NAME="thesis-itmo.pdf"

if ! command -v tectonic >/dev/null 2>&1; then
  echo "tectonic is required. Install it with: brew install tectonic" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE_DIR}/thesis-itmo.tex" ]]; then
  echo "Missing template. Run: git submodule update --init itmo-phd-thesis-template-en" >&2
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
  --exclude "${PDF_NAME}" \
  "${TEMPLATE_DIR}/" "${BUILD_DIR}/"

if [[ -d "${OVERLAY_DIR}" ]]; then
  rsync -a "${OVERLAY_DIR}/" "${BUILD_DIR}/"
fi

# Current polyglossia rejects babelshorthands for english; keep the upstream
# submodule untouched and patch only the disposable build copy.
perl -0pi -e 's/\\setmainlanguage\[babelshorthands=true\]\{english\}/\\setmainlanguage{english}/' \
  "${BUILD_DIR}/common/fonts.tex"
perl -0pi -e 's/\\cyrdash/---/g' \
  "${BUILD_DIR}/common/setupsimple.tex" \
  "${BUILD_DIR}/common/styles.tex"
perl -0pi -e 's/\\renewcommand\*\{\\cftchaptername\}\{\\chaptername\\space\}/\\renewcommand*{\\cftchaptername}{Глава\\space}/' \
  "${BUILD_DIR}/common/styles.tex"
perl -0pi -e 's/\\renewcommand\*\{\\cftappendixname\}\{\\appendixname\\space\}/\\renewcommand*{\\cftappendixname}{Приложение\\space}/' \
  "${BUILD_DIR}/common/styles.tex"

if [[ "${THESIS_FULL_BIB:-0}" != "1" ]]; then
  mkdir -p "${BUILD_DIR}/.bin"
  cat > "${BUILD_DIR}/.bin/biber" <<'EOF'
#!/usr/bin/env bash
: > thesis-itmo.bbl
exit 0
EOF
  chmod +x "${BUILD_DIR}/.bin/biber"
  export PATH="${BUILD_DIR}/.bin:${PATH}"
fi

(
  cd "${BUILD_DIR}"
  tectonic --keep-logs --keep-intermediates thesis-itmo.tex
)

cp "${BUILD_DIR}/${PDF_NAME}" "${OUT_DIR}/${PDF_NAME}"
echo "Wrote ${OUT_DIR}/${PDF_NAME}"
