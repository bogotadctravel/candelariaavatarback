module.exports = {
  apps: [
    {
      name: "api",
      script: "python3",
      args: "-m uvicorn main:app --host 0.0.0.0 --port 8001",
      interpreter: "none",
      watch: false
    },
    {
      name: "avataridt",
      script: "python3",
      args: "avatar.py start",
      interpreter: "none",
      watch: false
    }
  ]
};
