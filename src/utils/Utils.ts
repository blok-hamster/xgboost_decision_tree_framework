/**
 * @fileoverview General utility functions for the decision tree model toolkit
 * 
 * Provides helper functions for data manipulation, validation, type checking,
 * and common operations used throughout the library.
 */

import { Environment } from '../types';

/**
 * Utils class with static helper methods
 * 
 * Features:
 * - Data validation and type checking
 * - Array and object manipulation
 * - Statistical functions
 * - Random number generation with seeding
 * - Environment detection
 * - Performance measurement
 * 
 * @example
 * ```typescript
 * // Shuffle array
 * const shuffled = Utils.shuffleArray([1, 2, 3, 4, 5], 42);
 * 
 * // Check if numeric
 * const isNum = Utils.isNumeric('123.45'); // true
 * 
 * // Deep clone object
 * const cloned = Utils.deepClone(originalObject);
 * ```
 */
export class Utils {
  /**
   * Shuffles an array using Fisher-Yates algorithm with optional seeding
   * 
   * @param array - Array to shuffle
   * @param seed - Optional seed for reproducible shuffling
   * @returns Shuffled copy of the array
   * 
   * @example
   * ```typescript
   * const original = [1, 2, 3, 4, 5];
   * const shuffled = Utils.shuffleArray(original, 42);
   * // Returns a shuffled copy, original unchanged
   * ```
   */
  public static shuffleArray<T>(array: T[], seed?: number): T[] {
    const shuffled = [...array];
    const random = seed !== undefined ? this.createSeededRandom(seed) : Math.random;
    
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
    }
    
    return shuffled;
  }

  /**
   * Creates a seeded random number generator
   * 
   * @param seed - Seed value for reproducible random numbers
   * @returns Function that generates random numbers [0, 1)
   * 
   * @example
   * ```typescript
   * const random = Utils.createSeededRandom(42);
   * const num1 = random(); // Always the same for seed 42
   * const num2 = random();
   * ```
   */
  public static createSeededRandom(seed: number): () => number {
    let currentSeed = seed % 2147483647;
    if (currentSeed <= 0) currentSeed += 2147483646;
    
    return (): number => {
      currentSeed = (currentSeed * 16807) % 2147483647;
      return (currentSeed - 1) / 2147483646;
    };
  }

  /**
   * Checks if a value is numeric (number or numeric string)
   * 
   * @param value - Value to check
   * @returns Whether the value is numeric
   * 
   * @example
   * ```typescript
   * Utils.isNumeric(123);       // true
   * Utils.isNumeric('123.45');  // true
   * Utils.isNumeric('abc');     // false
   * Utils.isNumeric(null);      // false
   * ```
   */
  public static isNumeric(value: any): boolean {
    if (value === null || value === undefined) {
      return false;
    }
    
    if (typeof value === 'number') {
      return !isNaN(value) && isFinite(value);
    }
    
    if (typeof value === 'string') {
      return !isNaN(Number(value)) && isFinite(Number(value)) && value.trim() !== '';
    }
    
    return false;
  }

  /**
   * Safely converts a value to a number
   * 
   * @param value - Value to convert
   * @param defaultValue - Default value if conversion fails
   * @returns Converted number or default value
   * 
   * @example
   * ```typescript
   * Utils.toNumber('123.45', 0);    // 123.45
   * Utils.toNumber('abc', 0);       // 0
   * Utils.toNumber(null, -1);       // -1
   * ```
   */
  public static toNumber(value: any, defaultValue: number = 0): number {
    if (this.isNumeric(value)) {
      return Number(value);
    }
    return defaultValue;
  }

  /**
   * Deep clones an object or array
   * 
   * @param obj - Object to clone
   * @returns Deep copy of the object
   * 
   * @example
   * ```typescript
   * const original = { a: 1, b: { c: 2 } };
   * const cloned = Utils.deepClone(original);
   * cloned.b.c = 3; // Original remains unchanged
   * ```
   */
  public static deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') {
      return obj;
    }
    
    if (obj instanceof Date) {
      return new Date(obj.getTime()) as unknown as T;
    }
    
    if (obj instanceof Array) {
      return obj.map(item => this.deepClone(item)) as unknown as T;
    }
    
    if (typeof obj === 'object') {
      const cloned = {} as T;
      for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
          cloned[key] = this.deepClone(obj[key]);
        }
      }
      return cloned;
    }
    
    return obj;
  }

  /**
   * Calculates descriptive statistics for an array of numbers
   * 
   * @param values - Array of numeric values
   * @returns Statistics object
   * 
   * @example
   * ```typescript
   * const stats = Utils.calculateStats([1, 2, 3, 4, 5]);
   * console.log(stats.mean);    // 3
   * console.log(stats.median);  // 3
   * console.log(stats.std);     // ~1.58
   * ```
   */
  public static calculateStats(values: number[]): {
    count: number;
    mean: number;
    median: number;
    mode: number;
    min: number;
    max: number;
    std: number;
    variance: number;
    q25: number;
    q75: number;
  } {
    if (values.length === 0) {
      return {
        count: 0, mean: 0, median: 0, mode: 0,
        min: 0, max: 0, std: 0, variance: 0,
        q25: 0, q75: 0
      };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const count = values.length;
    const sum = values.reduce((acc, val) => acc + val, 0);
    const mean = sum / count;
    
    // Median
    const median = count % 2 === 0
      ? (sorted[count / 2 - 1]! + sorted[count / 2]!) / 2
      : sorted[Math.floor(count / 2)]!;
    
    // Mode (most frequent value)
    const frequency = new Map<number, number>();
    values.forEach(val => {
      frequency.set(val, (frequency.get(val) || 0) + 1);
    });
    const mode = Array.from(frequency.entries())
      .reduce((a, b) => b[1] > a[1] ? b : a)[0];
    
    // Min and Max
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    // Variance and Standard Deviation
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
    const std = Math.sqrt(variance);
    
    // Quartiles
    const q25Index = Math.floor(count * 0.25);
    const q75Index = Math.floor(count * 0.75);
    const q25 = sorted[q25Index]!;
    const q75 = sorted[q75Index]!;
    
    return { count, mean, median, mode, min, max, std, variance, q25, q75 };
  }

  /**
   * Validates that an object has all required properties
   * 
   * @param obj - Object to validate
   * @param requiredProps - Array of required property names
   * @returns Validation result
   * 
   * @example
   * ```typescript
   * const result = Utils.validateObject(
   *   { name: 'John', age: 30 },
   *   ['name', 'age', 'email']
   * );
   * console.log(result.isValid);        // false
   * console.log(result.missingProps);   // ['email']
   * ```
   */
  public static validateObject(obj: any, requiredProps: string[]): {
    isValid: boolean;
    missingProps: string[];
    extraProps: string[];
  } {
    if (!obj || typeof obj !== 'object') {
      return {
        isValid: false,
        missingProps: requiredProps,
        extraProps: []
      };
    }
    
    const objProps = Object.keys(obj);
    const missingProps = requiredProps.filter(prop => !objProps.includes(prop));
    const extraProps = objProps.filter(prop => !requiredProps.includes(prop));
    
    return {
      isValid: missingProps.length === 0,
      missingProps,
      extraProps
    };
  }

  /**
   * Sanitizes a string for use as a filename
   * 
   * @param filename - Original filename
   * @returns Sanitized filename
   * 
   * @example
   * ```typescript
   * const safe = Utils.sanitizeFilename('my/file<name>.txt');
   * console.log(safe); // 'my_file_name_.txt'
   * ```
   */
  public static sanitizeFilename(filename: string): string {
    return filename
      .replace(/[<>:"/\\|?*]/g, '_')  // Replace invalid characters
      .replace(/\s+/g, '_')          // Replace spaces with underscores
      .replace(/_+/g, '_')           // Replace multiple underscores with single
      .replace(/^_+|_+$/g, '');      // Remove leading/trailing underscores
  }

  /**
   * Formats a number with appropriate units (K, M, B)
   * 
   * @param num - Number to format
   * @param decimals - Number of decimal places
   * @returns Formatted string
   * 
   * @example
   * ```typescript
   * Utils.formatNumber(1234);      // '1.2K'
   * Utils.formatNumber(1234567);   // '1.2M'
   * Utils.formatNumber(0.123, 3);  // '0.123'
   * ```
   */
  public static formatNumber(num: number, decimals: number = 1): string {
    if (Math.abs(num) < 1000) {
      return num.toString();
    }
    
    const units = ['', 'K', 'M', 'B', 'T'];
    let unitIndex = 0;
    let value = Math.abs(num);
    
    while (value >= 1000 && unitIndex < units.length - 1) {
      value /= 1000;
      unitIndex++;
    }
    
    const formatted = value.toFixed(decimals);
    return (num < 0 ? '-' : '') + formatted + units[unitIndex];
  }

  /**
   * Formats bytes to human-readable format
   * 
   * @param bytes - Number of bytes
   * @param decimals - Number of decimal places
   * @returns Formatted string
   * 
   * @example
   * ```typescript
   * Utils.formatBytes(1024);     // '1.0 KB'
   * Utils.formatBytes(1048576);  // '1.0 MB'
   * ```
   */
  public static formatBytes(bytes: number, decimals: number = 1): string {
    if (bytes === 0) return '0 B';
    
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    const k = 1024;
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + units[i];
  }

  /**
   * Measures execution time of a function
   * 
   * @param fn - Function to measure
   * @returns Execution time in milliseconds and function result
   * 
   * @example
   * ```typescript
   * const { duration, result } = Utils.measureTime(() => {
   *   return expensiveOperation();
   * });
   * console.log(`Operation took ${duration}ms`);
   * ```
   */
  public static measureTime<T>(fn: () => T): { duration: number; result: T } {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;
    
    return { duration, result };
  }

  /**
   * Measures execution time of an async function
   * 
   * @param fn - Async function to measure
   * @returns Promise resolving to execution time in milliseconds and function result
   * 
   * @example
   * ```typescript
   * const { duration, result } = await Utils.measureTimeAsync(async () => {
   *   return await expensiveAsyncOperation();
   * });
   * console.log(`Operation took ${duration}ms`);
   * ```
   */
  public static async measureTimeAsync<T>(fn: () => Promise<T>): Promise<{ duration: number; result: T }> {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;
    
    return { duration, result };
  }

  /**
   * Debounces a function to limit how often it can be called
   * 
   * @param func - Function to debounce
   * @param wait - Wait time in milliseconds
   * @returns Debounced function
   * 
   * @example
   * ```typescript
   * const debouncedSave = Utils.debounce((data) => {
   *   saveToServer(data);
   * }, 500);
   * ```
   */
  public static debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout | number | undefined;
    
    return (...args: Parameters<T>) => {
      clearTimeout(timeout as any);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  /**
   * Detects the current runtime environment
   * 
   * @returns Environment information
   */
  public static detectEnvironment(): Environment {
    const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
    const isBrowser = typeof window !== 'undefined';
    
    let nodeVersion: string | undefined;
    let browserType: string | undefined;
    
    if (isNode) {
      nodeVersion = process.versions.node;
    }
    
    if (isBrowser) {
      // Simple browser detection
      const userAgent = navigator.userAgent;
      if (userAgent.includes('Chrome')) {
        browserType = 'chrome';
      } else if (userAgent.includes('Firefox')) {
        browserType = 'firefox';
      } else if (userAgent.includes('Safari')) {
        browserType = 'safari';
      } else if (userAgent.includes('Edge')) {
        browserType = 'edge';
      } else {
        browserType = 'unknown';
      }
    }
    
    return {
      isNode: !!isNode,
      isBrowser,
      ...(nodeVersion && { nodeVersion }),
      ...(browserType && { browserType }),
    };
  }

  /**
   * Creates a range of numbers
   * 
   * @param start - Start value (inclusive)
   * @param end - End value (exclusive)
   * @param step - Step size (default: 1)
   * @returns Array of numbers
   * 
   * @example
   * ```typescript
   * Utils.range(0, 5);       // [0, 1, 2, 3, 4]
   * Utils.range(0, 10, 2);   // [0, 2, 4, 6, 8]
   * Utils.range(5, 0, -1);   // [5, 4, 3, 2, 1]
   * ```
   */
  public static range(start: number, end: number, step: number = 1): number[] {
    const result: number[] = [];
    
    if (step === 0) {
      throw new Error('Step cannot be zero');
    }
    
    if (step > 0) {
      for (let i = start; i < end; i += step) {
        result.push(i);
      }
    } else {
      for (let i = start; i > end; i += step) {
        result.push(i);
      }
    }
    
    return result;
  }

  /**
   * Groups array elements by a key function
   * 
   * @param array - Array to group
   * @param keyFn - Function to extract grouping key
   * @returns Map with grouped elements
   * 
   * @example
   * ```typescript
   * const data = [
   *   { category: 'A', value: 1 },
   *   { category: 'B', value: 2 },
   *   { category: 'A', value: 3 }
   * ];
   * const grouped = Utils.groupBy(data, item => item.category);
   * // Map { 'A' => [...], 'B' => [...] }
   * ```
   */
  public static groupBy<T, K>(array: T[], keyFn: (item: T) => K): Map<K, T[]> {
    const groups = new Map<K, T[]>();
    
    for (const item of array) {
      const key = keyFn(item);
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(item);
    }
    
    return groups;
  }

  /**
   * Removes duplicate elements from an array
   * 
   * @param array - Array with potential duplicates
   * @param keyFn - Optional function to extract comparison key
   * @returns Array with unique elements
   * 
   * @example
   * ```typescript
   * Utils.unique([1, 2, 2, 3, 1]);           // [1, 2, 3]
   * Utils.unique(objects, obj => obj.id);    // Unique by ID
   * ```
   */
  public static unique<T>(array: T[], keyFn?: (item: T) => any): T[] {
    if (!keyFn) {
      return Array.from(new Set(array));
    }
    
    const seen = new Set();
    return array.filter(item => {
      const key = keyFn(item);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  /**
   * Safely gets a nested property from an object
   * 
   * @param obj - Object to access
   * @param path - Property path (e.g., 'user.profile.name')
   * @param defaultValue - Default value if path doesn't exist
   * @returns Property value or default
   * 
   * @example
   * ```typescript
   * const obj = { user: { profile: { name: 'John' } } };
   * Utils.getNestedProperty(obj, 'user.profile.name');     // 'John'
   * Utils.getNestedProperty(obj, 'user.email', 'N/A');     // 'N/A'
   * ```
   */
  public static getNestedProperty(obj: any, path: string, defaultValue: any = undefined): any {
    const keys = path.split('.');
    let current = obj;
    
    for (const key of keys) {
      if (current === null || current === undefined || !(key in current)) {
        return defaultValue;
      }
      current = current[key];
    }
    
    return current;
  }

  /**
   * Retries a function with exponential backoff
   * 
   * @param fn - Function to retry
   * @param maxRetries - Maximum number of retries
   * @param baseDelay - Base delay in milliseconds
   * @returns Promise resolving to function result
   * 
   * @example
   * ```typescript
   * const result = await Utils.retry(async () => {
   *   return await fetchData();
   * }, 3, 1000);
   * ```
   */
  public static async retry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        
        if (attempt === maxRetries) {
          throw lastError;
        }
        
        const delay = baseDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError!;
  }
} 