/**
 * @fileoverview Logger utility for structured logging with different levels
 * 
 * Provides a flexible logging system with configurable levels, structured output,
 * and environment-specific formatting (Node.js vs Browser).
 */

import { Logger, LogLevel, Environment } from '../types';

/**
 * Default logger implementation with console output
 * 
 * Features:
 * - Configurable log levels
 * - Structured output with timestamps
 * - Environment detection (Node.js vs Browser)
 * - Colored output in development
 * - JSON formatting in production
 * 
 * @example
 * ```typescript
 * const logger = new DefaultLogger(LogLevel.INFO);
 * logger.info('Training started', { modelType: 'XGBoost', features: 10 });
 * logger.error('Training failed', { error: 'Invalid data' });
 * ```
 */
export class DefaultLogger implements Logger {
  private level: LogLevel;
  private environment: Environment;
  private enableColors: boolean;

  /**
   * Creates a new logger instance
   * 
   * @param level - Minimum log level to output
   * @param enableColors - Whether to use colored output (auto-detected if not provided)
   */
  constructor(level: LogLevel = LogLevel.INFO, enableColors?: boolean) {
    this.level = level;
    this.environment = this.detectEnvironment();
    this.enableColors = enableColors ?? (this.environment.isNode && process.env.NODE_ENV !== 'production');
  }

  /**
   * Logs debug messages
   * 
   * @param message - Log message
   * @param meta - Additional metadata
   */
  public debug(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      this.log(LogLevel.DEBUG, message, meta);
    }
  }

  /**
   * Logs info messages
   * 
   * @param message - Log message
   * @param meta - Additional metadata
   */
  public info(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.INFO)) {
      this.log(LogLevel.INFO, message, meta);
    }
  }

  /**
   * Logs warning messages
   * 
   * @param message - Log message
   * @param meta - Additional metadata
   */
  public warn(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.WARN)) {
      this.log(LogLevel.WARN, message, meta);
    }
  }

  /**
   * Logs error messages
   * 
   * @param message - Log message
   * @param meta - Additional metadata
   */
  public error(message: string, meta?: any): void {
    if (this.shouldLog(LogLevel.ERROR)) {
      this.log(LogLevel.ERROR, message, meta);
    }
  }

  /**
   * Sets the minimum log level
   * 
   * @param level - New minimum log level
   */
  public setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Gets the current log level
   * 
   * @returns Current log level
   */
  public getLevel(): LogLevel {
    return this.level;
  }

  /**
   * Checks if a log level should be output
   * 
   * @private
   * @param level - Log level to check
   * @returns Whether the level should be logged
   */
  private shouldLog(level: LogLevel): boolean {
    const levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR];
    const currentLevelIndex = levels.indexOf(this.level);
    const messageLevel = levels.indexOf(level);
    return messageLevel >= currentLevelIndex;
  }

  /**
   * Performs the actual logging
   * 
   * @private
   * @param level - Log level
   * @param message - Log message
   * @param meta - Additional metadata
   */
  private log(level: LogLevel, message: string, meta?: any): void {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level: level.toUpperCase(),
      message,
      ...(meta && { meta }),
    };

    if (this.enableColors) {
      // Colored output for development
      const coloredLevel = this.colorizeLevel(level);
      const coloredMessage = this.colorizeMessage(message, level);
      
      if (meta) {
        console.log(`[${timestamp}] ${coloredLevel} ${coloredMessage}`, meta);
      } else {
        console.log(`[${timestamp}] ${coloredLevel} ${coloredMessage}`);
      }
    } else {
      // JSON output for production
      console.log(JSON.stringify(logEntry));
    }
  }

  /**
   * Colorizes log level for console output
   * 
   * @private
   * @param level - Log level
   * @returns Colorized level string
   */
  private colorizeLevel(level: LogLevel): string {
    const colors = {
      [LogLevel.DEBUG]: '\x1b[36m', // Cyan
      [LogLevel.INFO]: '\x1b[32m',  // Green
      [LogLevel.WARN]: '\x1b[33m',  // Yellow
      [LogLevel.ERROR]: '\x1b[31m', // Red
    };
    
    const resetColor = '\x1b[0m';
    return `${colors[level]}${level.toUpperCase()}${resetColor}`;
  }

  /**
   * Colorizes log message for console output
   * 
   * @private
   * @param message - Log message
   * @param level - Log level
   * @returns Colorized message string
   */
  private colorizeMessage(message: string, level: LogLevel): string {
    if (level === LogLevel.ERROR) {
      return `\x1b[31m${message}\x1b[0m`; // Red
    } else if (level === LogLevel.WARN) {
      return `\x1b[33m${message}\x1b[0m`; // Yellow
    }
    return message;
  }

  /**
   * Detects the runtime environment
   * 
   * @private
   * @returns Environment information
   */
  private detectEnvironment(): Environment {
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
    const isBrowser = typeof window !== 'undefined';
    
    let nodeVersion: string | undefined;
    let browserType: string | undefined;
    
    if (isNode) {
      nodeVersion = process.versions.node;
    }
    
    if (isBrowser) {
      browserType = this.detectBrowser();
    }
    
    return {
      isNode: !!isNode,
      isBrowser,
      ...(nodeVersion && { nodeVersion }),
      ...(browserType && { browserType }),
    };
  }

  /**
   * Detects browser type from user agent
   * 
   * @private
   * @returns Browser type string
   */
  private detectBrowser(): string {
    if (typeof window === 'undefined') return 'unknown';
    
    const userAgent = window.navigator.userAgent;
    if (userAgent.includes('Chrome')) return 'chrome';
    if (userAgent.includes('Firefox')) return 'firefox';
    if (userAgent.includes('Safari')) return 'safari';
    if (userAgent.includes('Edge')) return 'edge';
    return 'unknown';
  }
}

/**
 * Silent logger that doesn't output anything
 * 
 * Useful for testing or when logging is not desired.
 */
export class SilentLogger implements Logger {
  public debug(_message: string, _meta?: any): void {
    // No-op
  }

  public info(_message: string, _meta?: any): void {
    // No-op
  }

  public warn(_message: string, _meta?: any): void {
    // No-op
  }

  public error(_message: string, _meta?: any): void {
    // No-op
  }
}

/**
 * Creates a logger with custom configuration
 * 
 * @param config - Logger configuration
 * @returns Configured logger instance
 * 
 * @example
 * ```typescript
 * const logger = createLogger({
 *   level: LogLevel.DEBUG,
 *   enableColors: true,
 *   silent: false,
 * });
 * ```
 */
export function createLogger(config: {
  level?: LogLevel;
  enableColors?: boolean;
  silent?: boolean;
}): Logger {
  if (config.silent) {
    return new SilentLogger();
  }

  return new DefaultLogger(config.level, config.enableColors);
}

/**
 * Default logger instance for general use
 */
export const defaultLogger = new DefaultLogger();

/**
 * Logger for development (debug level, colored output)
 */
export const devLogger = new DefaultLogger(LogLevel.DEBUG, true);

/**
 * Logger for production (info level, JSON output)
 */
export const prodLogger = new DefaultLogger(LogLevel.INFO, false);

/**
 * Silent logger for testing
 */
export const silentLogger = new SilentLogger(); 