/**
 * @fileoverview FeatureAnalyzer implementation for dataset analysis and feature specification
 * 
 * The FeatureAnalyzer class scans datasets to determine optimal encoding parameters
 * for categorical features, including unique value counts and suggested bucket counts
 * for hash encoding.
 */

import {
  RawDataRecord,
  FeatureSpecMap,
  FeatureAnalysisResult,
  Logger,
} from '../types';

/**
 * FeatureAnalyzer class for analyzing datasets and determining feature specifications
 * 
 * This class provides comprehensive analysis of datasets to:
 * - Count unique values for categorical features
 * - Calculate optimal bucket counts for hash encoding
 * - Compute statistics for numerical features
 * - Detect data quality issues
 * - Generate feature specifications for model training
 * 
 * @example
 * ```typescript
 * const analyzer = new FeatureAnalyzer();
 * const data = [
 *   { color: 'red', size: 'large', price: 100 },
 *   { color: 'blue', size: 'medium', price: 80 },
 *   // ... more data
 * ];
 * 
 * const result = analyzer.analyzeFeatures(data, ['color', 'size'], ['price']);
 * console.log(result.categoricalSpecs);
 * // { color: { k: 2, bucketCount: 4 }, size: { k: 2, bucketCount: 4 } }
 * ```
 */
export class FeatureAnalyzer {
  private logger: Logger | undefined;

  /**
   * Creates a new FeatureAnalyzer instance
   * 
   * @param logger - Optional logger for debugging and monitoring
   */
  constructor(logger?: Logger) {
    this.logger = logger;
  }

  /**
   * Analyzes a dataset to determine feature specifications
   * 
   * This method performs comprehensive analysis of both categorical and numerical features:
   * 1. Counts unique values for each categorical feature
   * 2. Calculates optimal bucket counts using PRD rules
   * 3. Computes statistics for numerical features
   * 4. Detects data quality issues
   * 5. Generates overall dataset statistics
   * 
   * @param data - Array of raw data records
   * @param categoricalFeatures - Names of categorical features to analyze
   * @param numericFeatures - Names of numerical features to analyze
   * @param targetFeature - Name of the target variable
   * @returns Complete feature analysis result
   * 
   * @throws Error if data is empty or features are not found
   * 
   * @example
   * ```typescript
   * const analyzer = new FeatureAnalyzer();
   * const result = analyzer.analyzeFeatures(
   *   data,
   *   ['brand', 'category', 'color'],
   *   ['price', 'rating', 'sales'],
   *   'target'
   * );
   * 
   * // Access categorical feature specs
   * console.log(result.categoricalSpecs.brand.k); // unique count
   * console.log(result.categoricalSpecs.brand.bucketCount); // suggested buckets
   * 
   * // Access numerical feature stats
   * console.log(result.numericStats.price.mean);
   * console.log(result.numericStats.price.std);
   * ```
   */
  public analyzeFeatures(
    data: RawDataRecord[],
    categoricalFeatures: string[],
    numericFeatures: string[] = [],
    targetFeature?: string
  ): FeatureAnalysisResult {
    this.logger?.info('Starting feature analysis', {
      dataSize: data.length,
      categoricalFeatures: categoricalFeatures.length,
      numericFeatures: numericFeatures.length,
    });

    // Validate input data
    this.validateInputData(data, categoricalFeatures, numericFeatures, targetFeature);

    // Analyze categorical features
    const categoricalSpecs = this.analyzeCategoricalFeatures(data, categoricalFeatures);

    // Analyze numerical features
    const numericStats = this.analyzeNumericFeatures(data, numericFeatures);

    // Analyze target classes if provided
    const targetClasses = targetFeature ? this.analyzeTargetClasses(data, targetFeature) : [];

    // Generate dataset statistics
    const datasetStats = this.generateDatasetStats(
      data,
      categoricalFeatures,
      numericFeatures,
      targetClasses
    );

    this.logger?.info('Feature analysis completed', {
      categoricalFeaturesAnalyzed: Object.keys(categoricalSpecs).length,
      numericFeaturesAnalyzed: Object.keys(numericStats).length,
      targetClasses: targetClasses.length,
    });

    return {
      categoricalSpecs,
      numericStats,
      datasetStats,
    };
  }

  /**
   * Analyzes categorical features to determine unique counts and bucket specifications
   * 
   * @private
   * @param data - Dataset to analyze
   * @param categoricalFeatures - Names of categorical features
   * @returns Feature specifications for categorical features
   */
  private analyzeCategoricalFeatures(
    data: RawDataRecord[],
    categoricalFeatures: string[]
  ): FeatureSpecMap {
    const specs: FeatureSpecMap = {};

    for (const feature of categoricalFeatures) {
      this.logger?.debug(`Analyzing categorical feature: ${feature}`);

      const uniqueValues = new Set<string>();
      let nullCount = 0;

      // Count unique values for this feature
      for (const record of data) {
        const value = record[feature];
        
        if (value === null || value === undefined) {
          nullCount++;
          uniqueValues.add('__NULL__'); // Count nulls as a unique value
        } else {
          uniqueValues.add(String(value));
        }
      }

      const k = uniqueValues.size;
      const bucketCount = this.calculateBucketCount(k);

      // Warn about potential issues
      if (nullCount > data.length * 0.1) {
        this.logger?.warn(`Feature ${feature} has ${nullCount}/${data.length} null values (${(nullCount/data.length*100).toFixed(1)}%)`);
      }

      if (k === 1) {
        this.logger?.warn(`Feature ${feature} has only one unique value - consider removing it`);
      }

      if (k > 10000) {
        this.logger?.warn(`Feature ${feature} has ${k} unique values - very high cardinality may cause issues`);
      }

      specs[feature] = { k, bucketCount };

      this.logger?.debug(`Feature ${feature}: k=${k}, bucketCount=${bucketCount}`);
    }

    return specs;
  }

  /**
   * Analyzes numerical features to compute statistics
   * 
   * @private
   * @param data - Dataset to analyze
   * @param numericFeatures - Names of numerical features
   * @returns Statistics for numerical features
   */
  private analyzeNumericFeatures(
    data: RawDataRecord[],
    numericFeatures: string[]
  ): FeatureAnalysisResult['numericStats'] {
    const stats: FeatureAnalysisResult['numericStats'] = {};

    for (const feature of numericFeatures) {
      this.logger?.debug(`Analyzing numeric feature: ${feature}`);

      const values: number[] = [];
      let nullCount = 0;

      // Extract numeric values
      for (const record of data) {
        const value = record[feature];
        
        if (value === null || value === undefined) {
          nullCount++;
        } else {
          const numericValue = Number(value);
          if (!isNaN(numericValue)) {
            values.push(numericValue);
          } else {
            nullCount++;
          }
        }
      }

      if (values.length === 0) {
        this.logger?.warn(`Feature ${feature} has no valid numeric values`);
        stats[feature] = {
          min: 0,
          max: 0,
          mean: 0,
          std: 0,
          nullCount,
        };
        continue;
      }

      // Calculate statistics
      const min = Math.min(...values);
      const max = Math.max(...values);
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance);

      stats[feature] = {
        min,
        max,
        mean,
        std,
        nullCount,
      };

      // Warn about potential issues
      if (nullCount > data.length * 0.1) {
        this.logger?.warn(`Feature ${feature} has ${nullCount}/${data.length} null values (${(nullCount/data.length*100).toFixed(1)}%)`);
      }

      if (std === 0) {
        this.logger?.warn(`Feature ${feature} has zero standard deviation - all values are the same`);
      }

      this.logger?.debug(`Feature ${feature}: min=${min}, max=${max}, mean=${mean.toFixed(2)}, std=${std.toFixed(2)}`);
    }

    return stats;
  }

  /**
   * Analyzes target classes to determine unique class labels
   * 
   * @private
   * @param data - Dataset to analyze
   * @param targetFeature - Name of the target variable
   * @returns Array of unique class labels
   */
  private analyzeTargetClasses(data: RawDataRecord[], targetFeature: string): string[] {
    const uniqueClasses = new Set<string>();
    let nullCount = 0;

    for (const record of data) {
      const value = record[targetFeature];
      
      if (value === null || value === undefined) {
        nullCount++;
      } else {
        uniqueClasses.add(String(value));
      }
    }

    if (nullCount > 0) {
      this.logger?.warn(`Target feature ${targetFeature} has ${nullCount}/${data.length} null values`);
    }

    const classes = Array.from(uniqueClasses).sort();
    
    this.logger?.info(`Target feature ${targetFeature} has ${classes.length} unique classes:`, classes);

    return classes;
  }

  /**
   * Generates overall dataset statistics
   * 
   * @private
   * @param data - Dataset to analyze
   * @param categoricalFeatures - Names of categorical features
   * @param numericFeatures - Names of numerical features
   * @param targetClasses - Unique target classes
   * @returns Dataset statistics
   */
  private generateDatasetStats(
    data: RawDataRecord[],
    categoricalFeatures: string[],
    numericFeatures: string[],
    targetClasses: string[]
  ): FeatureAnalysisResult['datasetStats'] {
    let nullValues = 0;

    // Count null values across all features
    for (const record of data) {
      for (const feature of [...categoricalFeatures, ...numericFeatures]) {
        const value = record[feature];
        if (value === null || value === undefined) {
          nullValues++;
        }
      }
    }

    return {
      totalRows: data.length,
      totalFeatures: categoricalFeatures.length + numericFeatures.length,
      categoricalFeatures: categoricalFeatures.length,
      numericFeatures: numericFeatures.length,
      targetClasses,
      nullValues,
    };
  }

  /**
   * Calculates optimal bucket count based on unique values using PRD rules
   * 
   * Implements the rules from the PRD:
   * - If k ≤ 1000: bucketCount = k²
   * - Otherwise: bucketCount = 20 × k
   * 
   * @private
   * @param k - Number of unique values
   * @returns Optimal bucket count
   */
  private calculateBucketCount(k: number): number {
    if (k <= 1000) {
      return k * k;
    } else {
      return 20 * k;
    }
  }

  /**
   * Validates input data and features
   * 
   * @private
   * @param data - Dataset to validate
   * @param categoricalFeatures - Categorical feature names
   * @param numericFeatures - Numerical feature names
   * @param targetFeature - Target feature name
   * @throws Error if validation fails
   */
  private validateInputData(
    data: RawDataRecord[],
    categoricalFeatures: string[],
    numericFeatures: string[],
    targetFeature?: string
  ): void {
    if (!data || data.length === 0) {
      throw new Error('Data array is empty or undefined');
    }

    if (categoricalFeatures.length === 0 && numericFeatures.length === 0) {
      throw new Error('No features specified for analysis');
    }

    // Check if all features exist in the data
    const allFeatures = [...categoricalFeatures, ...numericFeatures];
    if (targetFeature) {
      allFeatures.push(targetFeature);
    }

    const sampleRecord = data[0];
    if (!sampleRecord) {
      throw new Error('First data record is undefined');
    }

    const missingFeatures = allFeatures.filter(feature => !(feature in sampleRecord));
    if (missingFeatures.length > 0) {
      throw new Error(`Features not found in data: ${missingFeatures.join(', ')}`);
    }

    // Check for duplicate feature names
    const duplicateFeatures = categoricalFeatures.filter(feature => 
      numericFeatures.includes(feature)
    );
    if (duplicateFeatures.length > 0) {
      throw new Error(`Features listed in both categorical and numeric: ${duplicateFeatures.join(', ')}`);
    }
  }

  /**
   * Suggests feature types based on data content
   * 
   * Analyzes the first few rows to suggest whether features should be treated
   * as categorical or numerical based on their content.
   * 
   * @param data - Dataset to analyze
   * @param sampleSize - Number of rows to sample (default: 100)
   * @returns Suggested feature categorization
   * 
   * @example
   * ```typescript
   * const analyzer = new FeatureAnalyzer();
   * const suggestions = analyzer.suggestFeatureTypes(data);
   * console.log(suggestions.categorical); // ['color', 'brand']
   * console.log(suggestions.numeric); // ['price', 'rating']
   * ```
   */
  public suggestFeatureTypes(
    data: RawDataRecord[],
    sampleSize: number = 100
  ): { categorical: string[]; numeric: string[] } {
    if (data.length === 0) {
      return { categorical: [], numeric: [] };
    }

    const sample = data.slice(0, Math.min(sampleSize, data.length));
    const features = Object.keys(sample[0] || {});
    const categorical: string[] = [];
    const numeric: string[] = [];

    for (const feature of features) {
      let numericCount = 0;
      let totalValid = 0;

      for (const record of sample) {
        const value = record[feature];
        
        if (value !== null && value !== undefined) {
          totalValid++;
          
          if (typeof value === 'number' || 
              (typeof value === 'string' && !isNaN(Number(value)))) {
            numericCount++;
          }
        }
      }

      // Classify based on majority type
      if (totalValid === 0) {
        // Skip features with all null values
        continue;
      } else if (numericCount / totalValid > 0.8) {
        numeric.push(feature);
      } else {
        categorical.push(feature);
      }
    }

    this.logger?.info('Feature type suggestions:', {
      categorical: categorical.length,
      numeric: numeric.length,
    });

    return { categorical, numeric };
  }

  /**
   * Analyzes feature correlation for numerical features
   * 
   * @param data - Dataset to analyze
   * @param numericFeatures - Names of numerical features
   * @returns Correlation matrix
   */
  public analyzeCorrelation(
    data: RawDataRecord[],
    numericFeatures: string[]
  ): { [feature1: string]: { [feature2: string]: number } } {
    const correlations: { [feature1: string]: { [feature2: string]: number } } = {};

    for (const feature1 of numericFeatures) {
      correlations[feature1] = {};
      
      for (const feature2 of numericFeatures) {
        if (feature1 === feature2) {
          correlations[feature1]![feature2] = 1.0;
        } else {
          const correlation = this.calculateCorrelation(data, feature1, feature2);
          correlations[feature1]![feature2] = correlation;
        }
      }
    }

    return correlations;
  }

  /**
   * Calculates Pearson correlation coefficient between two numerical features
   * 
   * @private
   * @param data - Dataset
   * @param feature1 - First feature name
   * @param feature2 - Second feature name
   * @returns Correlation coefficient (-1 to 1)
   */
  private calculateCorrelation(data: RawDataRecord[], feature1: string, feature2: string): number {
    const pairs: [number, number][] = [];

    for (const record of data) {
      const val1 = Number(record[feature1]);
      const val2 = Number(record[feature2]);
      
      if (!isNaN(val1) && !isNaN(val2)) {
        pairs.push([val1, val2]);
      }
    }

    if (pairs.length < 2) {
      return 0;
    }

    const mean1 = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
    const mean2 = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;

    let numerator = 0;
    let sum1Sq = 0;
    let sum2Sq = 0;

    for (const [val1, val2] of pairs) {
      const diff1 = val1 - mean1;
      const diff2 = val2 - mean2;
      
      numerator += diff1 * diff2;
      sum1Sq += diff1 * diff1;
      sum2Sq += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1Sq * sum2Sq);
    return denominator === 0 ? 0 : numerator / denominator;
  }
} 