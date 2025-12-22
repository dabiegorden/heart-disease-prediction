/**
 * Python Script Execution Utility
 */

import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { detectPythonCommand } from "./detect-python-command.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function executePythonScript(scriptName, args = [], options = {}) {
  const { onProgress } = options;
  const pythonCommand = await detectPythonCommand();
  const backendRoot = path.resolve(__dirname, "../..");
  const scriptPath = path.join(backendRoot, "python", scriptName);

  console.log(`[v0] Backend root directory: ${backendRoot}`);
  console.log(`[v0] Looking for script at: ${scriptPath}`);
  console.log(`[v0] Executing Python script: ${scriptPath}`);
  console.log(`[v0] Using Python command: ${pythonCommand}`);
  console.log(`[v0] With arguments:`, args);

  return new Promise((resolve, reject) => {
    const python = spawn(pythonCommand, [scriptPath, ...args], {
      cwd: backendRoot,
    });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data) => {
      const output = data.toString();
      stdout += output;
      console.log(`[Python stdout] ${output.trim()}`);

      if (onProgress) {
        const lines = output.split("\n");
        for (const line of lines) {
          if (line.startsWith("PROGRESS:")) {
            // Format: PROGRESS:percentage|message
            const [progressPart, ...messageParts] = line
              .substring(9)
              .split("|");
            const percentage = Number.parseInt(progressPart);
            const message = messageParts.join("|").trim();

            if (!isNaN(percentage)) {
              onProgress(percentage, message);
            }
          }
        }
      }
    });

    python.stderr.on("data", (data) => {
      const error = data.toString();
      stderr += error;
      console.error(`[Python stderr] ${error.trim()}`);
    });

    python.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}\n${stderr}`));
      } else {
        resolve({ stdout, stderr, exitCode: code });
      }
    });

    python.on("error", (err) => {
      reject(new Error(`Failed to start Python process: ${err.message}`));
    });
  });
}
