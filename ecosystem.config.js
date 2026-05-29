module.exports = {
  apps: [
    {
      name: "ws-avataridt",
      script: "./venv/bin/python",
      args: "-m fastapi run main.py --host 0.0.0.0 --port 8001",
      watch: false
    },
    {
      name: "avataridt",
      script: "./venv/bin/python",
      args: "avatar.py start",
      watch: false
    }
  ]
};
