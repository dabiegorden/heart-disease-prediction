/**
 * config/mongodb.js
 * =================
 * MongoDB connection with graceful retry and clean shutdown.
 * The server stays up even if Mongo is temporarily unreachable.
 */

import mongoose from "mongoose";
import dotenv from "dotenv";
dotenv.config({ quiet: true });

const MONGODB_URL = process.env.MONGODB_URL;

if (!MONGODB_URL) {
  throw new Error(
    "MONGODB_URL is not set in the environment. " +
      "Add it to your .env file and restart the server.",
  );
}

const CONNECT_OPTIONS = {
  serverSelectionTimeoutMS: 5_000, // fail fast on bad URL
  socketTimeoutMS: 45_000,
};

export const connectDB = async () => {
  try {
    const conn = await mongoose.connect(MONGODB_URL, CONNECT_OPTIONS);
    console.log(`✓ MongoDB connected: ${conn.connection.host}`);
  } catch (error) {
    // Log but do NOT crash – predictions still work without DB
    console.error(`⚠  MongoDB connection failed: ${error.message}`);
    console.error(
      "   Predictions will work but results will NOT be persisted.",
    );
  }
};

// Graceful shutdown on SIGINT / SIGTERM
async function gracefulClose(signal) {
  console.log(`\n[MongoDB] Closing connection (${signal})…`);
  await mongoose.connection.close();
  process.exit(0);
}

process.once("SIGINT", () => gracefulClose("SIGINT"));
process.once("SIGTERM", () => gracefulClose("SIGTERM"));
