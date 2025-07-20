/**
 * @fileoverview Unit tests for HashEncoder class
 */

import { HashEncoder } from '../src/core/HashEncoder';

describe('HashEncoder', () => {
  describe('Constructor', () => {
    it('should create an encoder with valid parameters', () => {
      const encoder = new HashEncoder(100, 42, 'test_feature');
      expect(encoder).toBeInstanceOf(HashEncoder);
      expect(encoder.getDimension()).toBe(100);
    });

    it('should use default parameters when not provided', () => {
      const encoder = new HashEncoder(50);
      expect(encoder.getDimension()).toBe(50);
    });

    it('should throw error for invalid bucket count', () => {
      expect(() => new HashEncoder(0)).toThrow('bucketCount must be a positive integer');
      expect(() => new HashEncoder(-1)).toThrow('bucketCount must be a positive integer');
      expect(() => new HashEncoder(1.5)).toThrow('bucketCount must be a positive integer');
    });

    it('should warn for very large bucket counts', () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      new HashEncoder(2000000);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('bucketCount 2000000 is very large')
      );
      consoleSpy.mockRestore();
    });
  });

  describe('encode()', () => {
    let encoder: HashEncoder;

    beforeEach(() => {
      encoder = new HashEncoder(10, 42, 'test');
    });

    it('should return Float32Array of correct length', () => {
      const result = encoder.encode('test');
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10);
    });

    it('should return one-hot encoded vector', () => {
      const result = encoder.encode('test');
      const nonZeroCount = Array.from(result).filter(val => val !== 0).length;
      const oneCount = Array.from(result).filter(val => val === 1).length;
      
      expect(nonZeroCount).toBe(1);
      expect(oneCount).toBe(1);
    });

    it('should be deterministic', () => {
      const result1 = encoder.encode('test');
      const result2 = encoder.encode('test');
      
      expect(Array.from(result1)).toEqual(Array.from(result2));
    });

    it('should produce different encodings for different values', () => {
      const result1 = encoder.encode('value1');
      const result2 = encoder.encode('value2');
      
      expect(Array.from(result1)).not.toEqual(Array.from(result2));
    });

    it('should handle string values', () => {
      const result = encoder.encode('hello world');
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10);
    });

    it('should handle number values', () => {
      const result = encoder.encode(123);
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10);
    });

    it('should handle boolean values', () => {
      const resultTrue = encoder.encode(true);
      const resultFalse = encoder.encode(false);
      
      expect(resultTrue).toBeInstanceOf(Float32Array);
      expect(resultFalse).toBeInstanceOf(Float32Array);
      expect(Array.from(resultTrue)).not.toEqual(Array.from(resultFalse));
    });

    it('should handle null and undefined values', () => {
      const resultNull = encoder.encode(null);
      const resultUndefined = encoder.encode(undefined);
      
      expect(resultNull).toBeInstanceOf(Float32Array);
      expect(resultUndefined).toBeInstanceOf(Float32Array);
      expect(Array.from(resultNull)).toEqual(Array.from(resultUndefined));
    });

    it('should handle empty string', () => {
      const result = encoder.encode('');
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10);
    });

    it('should handle special numeric values', () => {
      const resultNaN = encoder.encode(NaN);
      const resultInfinity = encoder.encode(Infinity);
      const resultNegInfinity = encoder.encode(-Infinity);
      
      expect(resultNaN).toBeInstanceOf(Float32Array);
      expect(resultInfinity).toBeInstanceOf(Float32Array);
      expect(resultNegInfinity).toBeInstanceOf(Float32Array);
      
      // All should be different
      expect(Array.from(resultNaN)).not.toEqual(Array.from(resultInfinity));
      expect(Array.from(resultInfinity)).not.toEqual(Array.from(resultNegInfinity));
    });
  });

  describe('encodeBatch()', () => {
    let encoder: HashEncoder;

    beforeEach(() => {
      encoder = new HashEncoder(10, 42);
    });

    it('should encode multiple values', () => {
      const values = ['a', 'b', 'c'];
      const results = encoder.encodeBatch(values);
      
      expect(results).toHaveLength(3);
      results.forEach(result => {
        expect(result).toBeInstanceOf(Float32Array);
        expect(result.length).toBe(10);
      });
    });

    it('should be consistent with single encode calls', () => {
      const values = ['test1', 'test2'];
      const batchResults = encoder.encodeBatch(values);
      const singleResults = values.map(v => encoder.encode(v));
      
      expect(batchResults).toHaveLength(singleResults.length);
      for (let i = 0; i < batchResults.length; i++) {
        expect(Array.from(batchResults[i])).toEqual(Array.from(singleResults[i]));
      }
    });

    it('should handle empty array', () => {
      const results = encoder.encodeBatch([]);
      expect(results).toHaveLength(0);
    });
  });

  describe('getConfig()', () => {
    it('should return correct configuration', () => {
      const encoder = new HashEncoder(100, 123, 'feature_name');
      const config = encoder.getConfig();
      
      expect(config).toEqual({
        bucketCount: 100,
        hashSeed: 123,
        featureName: 'feature_name',
      });
    });
  });

  describe('fromConfig()', () => {
    it('should create encoder from configuration', () => {
      const config = {
        bucketCount: 50,
        hashSeed: 456,
        featureName: 'test_feature',
      };
      
      const encoder = HashEncoder.fromConfig(config);
      expect(encoder.getDimension()).toBe(50);
      
      const encoderConfig = encoder.getConfig();
      expect(encoderConfig).toEqual(config);
    });
  });

  describe('estimateCollisionProbability()', () => {
    it('should return 0 for single unique value', () => {
      expect(HashEncoder.estimateCollisionProbability(1, 100)).toBe(0);
    });

    it('should return 1 when unique values >= bucket count', () => {
      expect(HashEncoder.estimateCollisionProbability(100, 50)).toBe(1);
      expect(HashEncoder.estimateCollisionProbability(100, 100)).toBe(1);
    });

    it('should return probability between 0 and 1 for normal cases', () => {
      const prob = HashEncoder.estimateCollisionProbability(50, 1000);
      expect(prob).toBeGreaterThan(0);
      expect(prob).toBeLessThan(1);
    });

    it('should increase with more unique values', () => {
      const prob1 = HashEncoder.estimateCollisionProbability(10, 100);
      const prob2 = HashEncoder.estimateCollisionProbability(20, 100);
      expect(prob2).toBeGreaterThan(prob1);
    });
  });

  describe('suggestBucketCount()', () => {
    it('should use k² for k <= 1000', () => {
      expect(HashEncoder.suggestBucketCount(10)).toBe(100);
      expect(HashEncoder.suggestBucketCount(50)).toBe(2500);
      expect(HashEncoder.suggestBucketCount(1000)).toBe(1000000);
    });

    it('should use 20×k for k > 1000', () => {
      expect(HashEncoder.suggestBucketCount(1001)).toBe(20020);
      expect(HashEncoder.suggestBucketCount(2000)).toBe(40000);
    });

    it('should return 1 for empty or single value', () => {
      expect(HashEncoder.suggestBucketCount(0)).toBe(1);
      expect(HashEncoder.suggestBucketCount(1)).toBe(1);
    });
  });

  describe('validateDeterminism()', () => {
    it('should return true for deterministic encoding', () => {
      const encoder = new HashEncoder(100, 42);
      expect(encoder.validateDeterminism('test')).toBe(true);
    });

    it('should validate across multiple iterations', () => {
      const encoder = new HashEncoder(50, 123);
      expect(encoder.validateDeterminism('test_value', 200)).toBe(true);
    });

    it('should work with different value types', () => {
      const encoder = new HashEncoder(25, 456);
      expect(encoder.validateDeterminism('string')).toBe(true);
      expect(encoder.validateDeterminism(12345)).toBe(true);
      expect(encoder.validateDeterminism(true)).toBe(true);
    });
  });

  describe('toString()', () => {
    it('should return meaningful string representation', () => {
      const encoder = new HashEncoder(100, 42, 'color');
      const str = encoder.toString();
      
      expect(str).toContain('HashEncoder');
      expect(str).toContain('bucketCount=100');
      expect(str).toContain('hashSeed=42');
      expect(str).toContain('feature=color');
    });
  });

  describe('Different seeds produce different encodings', () => {
    it('should produce different results with different seeds', () => {
      const encoder1 = new HashEncoder(100, 42);
      const encoder2 = new HashEncoder(100, 123);
      
      const result1 = encoder1.encode('test');
      const result2 = encoder2.encode('test');
      
      expect(Array.from(result1)).not.toEqual(Array.from(result2));
    });
  });

  describe('Edge cases', () => {
    it('should handle very long strings', () => {
      const encoder = new HashEncoder(100, 42);
      const longString = 'a'.repeat(10000);
      const result = encoder.encode(longString);
      
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(100);
    });

    it('should handle unicode characters', () => {
      const encoder = new HashEncoder(50, 42);
      const unicodeString = '你好世界🌍';
      const result = encoder.encode(unicodeString);
      
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(50);
    });

    it('should handle whitespace properly', () => {
      const encoder = new HashEncoder(20, 42);
      
      const result1 = encoder.encode('  test  ');
      const result2 = encoder.encode('test');
      
      // Should be the same after trimming
      expect(Array.from(result1)).toEqual(Array.from(result2));
    });
  });
}); 