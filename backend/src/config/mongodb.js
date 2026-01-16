import mongoose from "mongoose";
import dotenv from "dotenv";
dotenv.config({ quiet: true });

const MONGODB_URL = process.env.MONGODB_URL;

if (!MONGODB_URL) {
  throw new Error("Please provide the MONGODB_URL in env file.");
}

export const connectDB = async () => {
  try {
    const conn = await mongoose.connect(MONGODB_URL);
    console.log(`MongoDB connected: ${conn.connection.host}`);
  } catch (error) {
    console.log(`Error connecting to the database: ${error}`);
  }
};
