#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — cài SBD API lên máy chủ Linux (Ubuntu/Debian)
#
# Chạy với quyền root:
#   sudo bash deploy.sh
#
# Sau khi chạy xong:
#   curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DEPLOY_DIR="/opt/sbd"
SERVICE_USER="sbd"
PORT=8000

echo "═══════════════════════════════════════════"
echo " SBD API — Deploy script"
echo " Target: ${DEPLOY_DIR}"
echo "═══════════════════════════════════════════"

# ── 1. Kiểm tra quyền root ───────────────────────────────────────────────────
if [[ "$EUID" -ne 0 ]]; then
  echo "❌ Chạy lại với sudo: sudo bash deploy.sh" >&2
  exit 1
fi

# ── 2. Cài Python + dependencies hệ thống ───────────────────────────────────
echo ""
echo "▶ Cài Python 3.11 + build deps..."
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev build-essential curl

# ── 3. Tạo user riêng cho service ───────────────────────────────────────────
echo ""
echo "▶ Tạo system user '${SERVICE_USER}'..."
if ! id "$SERVICE_USER" &>/dev/null; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
  echo "   User '${SERVICE_USER}' đã tạo."
else
  echo "   User '${SERVICE_USER}' đã tồn tại, bỏ qua."
fi

# ── 4. Copy project vào /opt/sbd ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "▶ Copy project từ ${SCRIPT_DIR} → ${DEPLOY_DIR}..."
mkdir -p "$DEPLOY_DIR"
rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
         --exclude='.env' --exclude='.git' \
         "${SCRIPT_DIR}/" "${DEPLOY_DIR}/"

# ── 5. Tạo virtualenv và cài deps ───────────────────────────────────────────
echo ""
echo "▶ Tạo virtualenv tại ${DEPLOY_DIR}/.venv..."
python3.11 -m venv "${DEPLOY_DIR}/.venv"

echo "▶ Cài dependencies (chỉ API server, không cần training deps)..."
"${DEPLOY_DIR}/.venv/bin/pip" install --quiet --upgrade pip
"${DEPLOY_DIR}/.venv/bin/pip" install --quiet \
  torch --index-url https://download.pytorch.org/whl/cpu
"${DEPLOY_DIR}/.venv/bin/pip" install --quiet \
  -r "${DEPLOY_DIR}/requirements.txt"

# ── 6. Phân quyền ───────────────────────────────────────────────────────────
echo ""
echo "▶ Phân quyền cho user '${SERVICE_USER}'..."
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$DEPLOY_DIR"
chmod -R 750 "$DEPLOY_DIR"

# ── 7. Cài systemd service ───────────────────────────────────────────────────
echo ""
echo "▶ Cài systemd service..."
cp "${DEPLOY_DIR}/sbd.service" /etc/systemd/system/sbd.service

# Cập nhật path trong service file
sed -i "s|/opt/sbd|${DEPLOY_DIR}|g" /etc/systemd/system/sbd.service
sed -i "s|User=sbd|User=${SERVICE_USER}|g" /etc/systemd/system/sbd.service
sed -i "s|Group=sbd|Group=${SERVICE_USER}|g" /etc/systemd/system/sbd.service

systemctl daemon-reload
systemctl enable sbd.service
systemctl restart sbd.service

# ── 8. Kiểm tra ─────────────────────────────────────────────────────────────
echo ""
echo "▶ Chờ service khởi động (30s cho model load)..."
sleep 30

if systemctl is-active --quiet sbd.service; then
  echo "✅ Service đang chạy!"
  curl -s "http://localhost:${PORT}/health" | python3 -m json.tool
else
  echo "❌ Service lỗi — xem log:"
  journalctl -u sbd.service -n 30 --no-pager
  exit 1
fi

echo ""
echo "═══════════════════════════════════════════"
echo " Deploy thành công!"
echo ""
echo " Endpoints:"
echo "   GET  http://localhost:${PORT}/"
echo "   GET  http://localhost:${PORT}/health"
echo "   POST http://localhost:${PORT}/predict"
echo "   POST http://localhost:${PORT}/predict/batch"
echo ""
echo " Quản lý service:"
echo "   systemctl status sbd"
echo "   systemctl restart sbd"
echo "   journalctl -u sbd -f    # xem log realtime"
echo ""
echo " Đổi threshold (không cần restart code):"
echo "   systemctl edit sbd      # thêm Environment=SBD_THRESHOLD=0.70"
echo "   systemctl restart sbd"
echo "═══════════════════════════════════════════"
