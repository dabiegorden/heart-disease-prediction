/**
 * Global Error Handling Middleware
 * Provides consistent, safe, and structured error responses.
 */

export function errorHandler(err, req, res, next) {
  const isDev = process.env.NODE_ENV !== "production";

  // Normalize status code
  const statusCode = err.statusCode || 500;

  // Normalize error message
  const message = err.message || "Internal Server Error";

  // Log errors only in development
  if (isDev) {
    console.error("❌ ERROR:", {
      message: err.message,
      stack: err.stack,
      path: req.path,
    });
  }

  res.status(statusCode).json({
    success: false,
    error: {
      message,
      type: err.name || "Error",
    },
    path: req.path,
    status: statusCode,
    timestamp: new Date().toISOString(),
  });
}

/**
 * Wrap async route handlers and forward errors automatically
 */
export function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}
