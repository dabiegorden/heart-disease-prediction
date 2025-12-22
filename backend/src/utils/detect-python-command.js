/**
 * Detect available Python command
 */

import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const PYTHON_COMMANDS = ["python3", "python", "py"];

export async function detectPythonCommand() {
  console.log("[v0] Detecting Python command...");

  for (const cmd of PYTHON_COMMANDS) {
    try {
      const { stdout } = await execAsync(`${cmd} --version`);
      const version = stdout.trim();
      console.log(`[v0] Detected Python command: ${cmd} (${version})`);
      return cmd;
    } catch (error) {
      // Command not found, try next one
      continue;
    }
  }

  console.error("[v0] No Python command found!");
  throw new Error("Python is not installed or not in PATH");
}
