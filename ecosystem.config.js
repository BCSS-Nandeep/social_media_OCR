// PM2 process definition. Usage: pm2 start ecosystem.config.js
// See DEPLOYMENT.md for the full setup (Node/PM2 install, startup, save).
module.exports = {
  apps: [
    {
      name: "social-media-ocr-api",
      script: "scripts/start_api.sh",
      interpreter: "none",   // the script has its own #!/usr/bin/env bash shebang
      cwd: __dirname,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      env: { PORT: "8000" },
    },
  ],
};
