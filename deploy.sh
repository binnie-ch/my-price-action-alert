#!/usr/bin/env bash
# One-shot setup on an Ubuntu/Debian server. Run from the bot folder:  bash deploy.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip
sudo timedatectl set-timezone Asia/Kolkata || true

python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install --upgrade pip
"$DIR/venv/bin/pip" install -r "$DIR/requirements.txt"

if [ ! -f "$DIR/.env" ]; then
  cat > "$DIR/.env" <<ENVFILE
ANGEL_API_KEY=your_smartapi_key
ANGEL_CLIENT_ID=your_angel_client_id
ANGEL_PIN=your_angel_pin
ANGEL_TOTP_SECRET=your_totp_secret
TELEGRAM_BOT_TOKEN=123456:ABC-your-bot-token
TELEGRAM_CHAT_ID=your_numeric_chat_id
ENVFILE
  chmod 600 "$DIR/.env"
  echo ""
  echo "Now fill in your keys:   nano $DIR/.env"
  echo "Then run again:          bash deploy.sh"
  exit 0
fi

echo "Sending Telegram test..."
set -a; . "$DIR/.env"; set +a
"$DIR/venv/bin/python" "$DIR/price_action_bot.py" --test || true

sudo tee /etc/systemd/system/pricebot.service > /dev/null <<UNIT
[Unit]
Description=Price action alert bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$DIR
EnvironmentFile=$DIR/.env
Environment=PYTHONUNBUFFERED=1
ExecStart=$DIR/venv/bin/python $DIR/price_action_bot.py
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable pricebot
sudo systemctl restart pricebot
echo ""
echo "Bot is running 24x7. Live logs:  sudo journalctl -u pricebot -f"
echo "Stop: sudo systemctl stop pricebot   |   Restart: sudo systemctl restart pricebot"
