import { spawn } from "node:child_process";

const isWindows = process.platform === "win32";
const npm = isWindows ? "npm.cmd" : "npm";

function run(name, args) {
  const child = spawn(npm, args, {
    stdio: "inherit",
    shell: false,
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      console.log(`${name} stopped by ${signal}`);
    } else if (code !== 0) {
      console.log(`${name} exited with code ${code}`);
    }
  });

  return child;
}

const backend = run("backend", ["--prefix", "backend", "run", "dev"]);
const frontend = run("frontend", ["--prefix", "frontend", "run", "dev"]);

function shutdown() {
  backend.kill();
  frontend.kill();
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
