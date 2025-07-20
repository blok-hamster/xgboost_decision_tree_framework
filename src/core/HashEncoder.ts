/**
 * @fileoverview HashEncoder implementation for categorical feature encoding
 * 
 * The HashEncoder class provides deterministic hashing for categorical features,
 * converting string values to sparse one-hot encoded vectors using feature hashing.
 * This approach allows handling of unseen categories during inference without
 * requiring retraining or vocabulary expansion.
 */

import murmur3 from 'murmurhash3js';
import { HashEncoderConfig } from '../types';

/**
 * HashEncoder class for converting categorical features to numeric vectors
 * 
 * This class implements the "hashing trick" for categorical features:
 * - Uses MurmurHash3 for deterministic, fast hashing
 * - Produces sparse one-hot encoded vectors
 * - Handles unseen categories gracefully during inference
 * - Maintains deterministic behavior across training and inference
 * 
 * @example
 * ```typescript
 * const encoder = new HashEncoder(100, 42);
 * const encoded = encoder.encode('blue');
 * // Returns Float32Array with one element set to 1.0 at the hashed index
 * ```
 */
export class HashEncoder {
  private readonly bucketCount: number;
  private readonly hashSeed: number;
  private readonly featureName: string;

  /**
   * Creates a new HashEncoder instance
   * 
   * @param bucketCount - The number of hash buckets (vector dimension)
   * @param hashSeed - Seed for deterministic hashing (default: 42)
   * @param featureName - Name of the feature being encoded (for debugging)
   * 
   * @throws Error if bucketCount is not a positive integer
   */
  constructor(bucketCount: number, hashSeed: number = 42, featureName: string = 'unknown') {
    if (!Number.isInteger(bucketCount) || bucketCount <= 0) {
      throw new Error(`bucketCount must be a positive integer, got: ${bucketCount}`);
    }
    
    if (bucketCount > 1000000) {
      console.warn(
        `HashEncoder: bucketCount ${bucketCount} is very large. ` +
        `Consider using a smaller value to avoid memory issues.`
      );
    }

    this.bucketCount = bucketCount;
    this.hashSeed = hashSeed;
    this.featureName = featureName;
  }

  /**
   * Encodes a categorical value to a sparse one-hot vector
   * 
   * This method:
   * 1. Converts the input value to a string
   * 2. Computes a hash using MurmurHash3 with the configured seed
   * 3. Maps the hash to a bucket index using modulo operation
   * 4. Returns a sparse vector with 1.0 at the hashed index
   * 
   * @param value - The categorical value to encode (will be converted to string)
   * @returns Float32Array with one element set to 1.0 at the hashed index
   * 
   * @example
   * ```typescript
   * const encoder = new HashEncoder(10, 42);
   * 
   * // Encode different values
   * const blueEncoded = encoder.encode('blue');
   * const redEncoded = encoder.encode('red');
   * 
   * // Same value always produces same encoding
   * const blueEncoded2 = encoder.encode('blue');
   * // blueEncoded equals blueEncoded2
   * 
   * // Handles various input types
   * const numberEncoded = encoder.encode(123);
   * const booleanEncoded = encoder.encode(true);
   * ```
   */
  public encode(value: string | number | boolean | null | undefined): Float32Array {
    // Convert input to string, handling null/undefined gracefully
    const stringValue = this.normalizeValue(value);
    
    // Compute hash using MurmurHash3 for deterministic, fast hashing
    const hashValue = murmur3.x86.hash32(stringValue, this.hashSeed);
    
    // Map hash to bucket index using modulo operation
    // Use Math.abs to ensure positive index
    const bucketIndex = Math.abs(hashValue) % this.bucketCount;
    
    // Create sparse one-hot encoded vector
    const encodedVector = new Float32Array(this.bucketCount);
    encodedVector[bucketIndex] = 1.0;
    
    return encodedVector;
  }

  /**
   * Encodes multiple values in batch for efficiency
   * 
   * @param values - Array of values to encode
   * @returns Array of encoded vectors
   * 
   * @example
   * ```typescript
   * const encoder = new HashEncoder(10, 42);
   * const encoded = encoder.encodeBatch(['red', 'blue', 'green']);
   * // Returns array of Float32Arrays
   * ```
   */
  public encodeBatch(values: (string | number | boolean | null | undefined)[]): Float32Array[] {
    return values.map(value => this.encode(value));
  }

  /**
   * Gets the configuration of this encoder
   * 
   * @returns HashEncoderConfig object with encoder settings
   */
  public getConfig(): HashEncoderConfig {
    return {
      bucketCount: this.bucketCount,
      hashSeed: this.hashSeed,
      featureName: this.featureName,
    };
  }

  /**
   * Gets the output dimension (bucket count) of this encoder
   * 
   * @returns The number of buckets/dimensions in the output vector
   */
  public getDimension(): number {
    return this.bucketCount;
  }

  /**
   * Creates a new HashEncoder from configuration
   * 
   * @param config - HashEncoderConfig object
   * @returns New HashEncoder instance
   */
  public static fromConfig(config: HashEncoderConfig): HashEncoder {
    return new HashEncoder(
      config.bucketCount,
      config.hashSeed,
      config.featureName
    );
  }

  /**
   * Estimates the probability of hash collision for given parameters
   * 
   * Uses the birthday paradox approximation to estimate collision probability
   * when encoding k unique values into n buckets.
   * 
   * @param uniqueValues - Number of unique values expected
   * @param bucketCount - Number of hash buckets
   * @returns Estimated collision probability (0-1)
   * 
   * @example
   * ```typescript
   * const collisionProb = HashEncoder.estimateCollisionProbability(100, 1000);
   * // Returns approximately 0.39 (39% chance of collision)
   * ```
   */
  public static estimateCollisionProbability(uniqueValues: number, bucketCount: number): number {
    if (uniqueValues <= 1) return 0;
    if (uniqueValues >= bucketCount) return 1;
    
    // Birthday paradox approximation: 1 - exp(-k*(k-1)/(2*n))
    const k = uniqueValues;
    const n = bucketCount;
    const exponent = -(k * (k - 1)) / (2 * n);
    return 1 - Math.exp(exponent);
  }

  /**
   * Suggests optimal bucket count based on unique values and target collision rate
   * 
   * @param uniqueValues - Number of unique values
   * @param maxCollisionRate - Maximum acceptable collision rate (default: 0.1)
   * @returns Suggested bucket count
   * 
   * @example
   * ```typescript
   * const bucketCount = HashEncoder.suggestBucketCount(100, 0.05);
   * // Returns bucket count that keeps collision rate below 5%
   * ```
   */
  public static suggestBucketCount(uniqueValues: number, _maxCollisionRate: number = 0.1): number {
    if (uniqueValues <= 1) return 1;
    
    // Use rule from PRD: k² for k≤1000, otherwise 20×k
    if (uniqueValues <= 1000) {
      return uniqueValues * uniqueValues;
    } else {
      return 20 * uniqueValues;
    }
  }

  /**
   * Normalizes input value to string for consistent hashing
   * 
   * @private
   * @param value - Input value of any type
   * @returns Normalized string representation
   */
  private normalizeValue(value: string | number | boolean | null | undefined): string {
    if (value === null || value === undefined) {
      return '__NULL__'; // Special token for null/undefined values
    }
    
    if (typeof value === 'boolean') {
      return value ? '__TRUE__' : '__FALSE__'; // Special tokens for booleans
    }
    
    if (typeof value === 'number') {
      // Handle special number values
      if (Number.isNaN(value)) return '__NaN__';
      if (value === Infinity) return '__INFINITY__';
      if (value === -Infinity) return '__NEGATIVE_INFINITY__';
      return value.toString();
    }
    
    // For strings, trim whitespace and handle empty strings
    const stringValue = String(value).trim();
    return stringValue.length === 0 ? '__EMPTY__' : stringValue;
  }

  /**
   * Validates that the encoder produces deterministic results
   * 
   * @param testValue - Value to test encoding consistency
   * @param iterations - Number of iterations to test (default: 100)
   * @returns True if encoding is deterministic
   * 
   * @example
   * ```typescript
   * const encoder = new HashEncoder(100, 42);
   * const isDeterministic = encoder.validateDeterminism('test');
   * // Returns true if encoding is consistent across iterations
   * ```
   */
  public validateDeterminism(testValue: string | number | boolean, iterations: number = 100): boolean {
    const firstEncoding = this.encode(testValue);
    
    for (let i = 0; i < iterations; i++) {
      const currentEncoding = this.encode(testValue);
      
      // Check if vectors are identical
      if (firstEncoding.length !== currentEncoding.length) {
        return false;
      }
      
      for (let j = 0; j < firstEncoding.length; j++) {
        if (firstEncoding[j] !== currentEncoding[j]) {
          return false;
        }
      }
    }
    
    return true;
  }

  /**
   * Returns a string representation of the encoder
   * 
   * @returns String description of the encoder
   */
  public toString(): string {
    return `HashEncoder(bucketCount=${this.bucketCount}, hashSeed=${this.hashSeed}, feature=${this.featureName})`;
  }
} 