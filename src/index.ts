/**
 * @fileoverview Main entry point for the XGBoost Decision Tree Toolkit
 * 
 * Exports all core components, utilities, and types for XGBoost-based
 * decision tree models with support for both classification and regression.
 */


// Core components
export { HashEncoder } from './core/HashEncoder';
export { FeatureAnalyzer } from './core/FeatureAnalyzer';
export { Trainer } from './core/Trainer';
export { Model } from './core/Model';
export { FeatureExtractor } from './core/FeatureExtractor';
export { TreeVisualizer } from './utils/TreeVisualizer';

// Utilities
export { DefaultLogger, SilentLogger, createLogger, defaultLogger, devLogger, prodLogger, silentLogger } from './utils/Logger';
export { DataLoader } from './utils/DataLoader';
export { Metrics } from './utils/Metrics';
export { Utils } from './utils/Utils';

// Types
export * from './types';

// Version info
export const VERSION = '1.0.0';

/**
 * Library information
 */
export const LIBRARY_INFO = {
  name: 'Decision Tree Model Toolkit',
  version: VERSION,
  description: 'A comprehensive TypeScript toolkit for XGBoost-based decision tree models',
  author: 'Decision Tree Model Team',
  license: 'MIT',
  repository: 'https://github.com/your-org/decision-tree-model',
};

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG = {
  // HashEncoder defaults
  hashEncoder: {
    defaultHashSeed: 42,
    maxBucketCountWarning: 1000000,
  },
  
  // FeatureAnalyzer defaults
  featureAnalyzer: {
    bucketCountMultiplier: {
      smallK: 1000,      // k <= 1000
      smallMultiplier: 1, // k^2
      largeMultiplier: 20, // 20 * k
    },
    highCardinalityWarning: 10000,
    highNullRateWarning: 0.1,
  },
  
  // Training defaults
  training: {
    defaultTestRatio: 0.2,
    defaultCrossValidationFolds: 5,
    defaultSeed: 42,
    performanceTarget: 100, // milliseconds per sample
  },
  
  // XGBoost defaults
  xgboost: {
    binaryClassification: {
      objective: 'binary:logistic',
      max_depth: 6,
      eta: 0.1,
      num_boost_round: 50,
      eval_metric: 'auc',
      seed: 42,
      booster: 'gbtree',
      subsample: 1.0,
      colsample_bytree: 1.0,
      verbosity: 1,
      min_child_weight: 1,
    },
    multiClassification: {
      objective: 'multi:softprob',
      max_depth: 6,
      eta: 0.1,
      num_boost_round: 50,
      eval_metric: 'mlogloss',
      seed: 42,
      booster: 'gbtree',
      subsample: 1.0,
      colsample_bytree: 1.0,
      verbosity: 1,
      min_child_weight: 1,
    },
    regression: {
      objective: 'reg:squarederror',
      max_depth: 5,
      eta: 0.05,
      num_boost_round: 100,
      eval_metric: 'rmse',
      seed: 42,
      booster: 'gbtree',
      subsample: 0.8,
      colsample_bytree: 0.8,
      verbosity: 1,
      min_child_weight: 1,
    },
  },
  
  // Data loading defaults
  dataLoading: {
    csv: {
      delimiter: ',',
      hasHeader: true,
      encoding: 'utf8',
    },
    qualityThresholds: {
      maxNullRate: 0.5,
      maxDuplicateRate: 0.1,
    },
  },
  
  // Logging defaults
  logging: {
    defaultLevel: 'info',
    enableColors: true,
    enableTimestamps: true,
  },
} as const;

/**
 * Quick-start configuration generator
 * 
 * @param type - Type of task ('binary', 'multiclass', or 'regression')
 * @returns Complete configuration object
 * 
 * @example
 * ```typescript
 * // Binary classification setup
 * const config = createQuickStartConfig('binary');
 * console.log(config.xgbParams); // XGBoost parameters optimized for binary classification
 * 
 * // Regression setup
 * const regressionConfig = createQuickStartConfig('regression');
 * console.log(regressionConfig.xgbParams); // XGBoost parameters optimized for regression
 * ```
 */
export function createQuickStartConfig(type: 'binary' | 'multiclass' | 'regression') {
  const baseConfig = {
    categoricalFeatures: ['category', 'brand', 'color'],
    numericFeatures: ['price', 'rating', 'sales'],
    target: type === 'regression' ? 'value' : 'class',
    taskType: type === 'regression' ? 'regression' : 'classification',
  };

  if (type === 'binary') {
    return {
      ...baseConfig,
      xgbParams: { ...DEFAULT_CONFIG.xgboost.binaryClassification },
    };
  } else if (type === 'multiclass') {
    return {
      ...baseConfig,
      xgbParams: { ...DEFAULT_CONFIG.xgboost.multiClassification },
    };
  } else {
    return {
      ...baseConfig,
      xgbParams: { ...DEFAULT_CONFIG.xgboost.regression },
    };
  }
}

/**
 * Creates a quick-start classification trainer
 * 
 * @param categoricalFeatures - Array of categorical feature names
 * @param numericFeatures - Array of numeric feature names
 * @param target - Target variable name
 * @param xgbParams - Optional XGBoost parameters
 * @returns Configured Trainer instance
 */
export function createClassificationTrainer(
  categoricalFeatures: string[],
  numericFeatures: string[],
  target: string,
  xgbParams?: any
) {
  const config = {
    categoricalFeatures,
    numericFeatures,
    target,
    taskType: 'classification' as const,
    xgbParams: { ...DEFAULT_CONFIG.xgboost.binaryClassification, ...xgbParams },
  };
  
  const { Trainer } = require('./core/Trainer');
  return new Trainer(config);
}

/**
 * Creates a quick-start regression trainer
 * 
 * @param categoricalFeatures - Array of categorical feature names
 * @param numericFeatures - Array of numeric feature names
 * @param target - Target variable name
 * @param xgbParams - Optional XGBoost parameters
 * @returns Configured Trainer instance
 */
export function createRegressionTrainer(
  categoricalFeatures: string[],
  numericFeatures: string[],
  target: string,
  xgbParams?: any
) {
  const config = {
    categoricalFeatures,
    numericFeatures,
    target,
    taskType: 'regression' as const,
    xgbParams: { ...DEFAULT_CONFIG.xgboost.regression, ...xgbParams },
  };
  
  const { Trainer } = require('./core/Trainer');
  return new Trainer(config);
}

/**
 * Validates model configuration for task type
 * 
 * @param config - Configuration object to validate
 * @returns Validation result
 */
export function validateModelConfig(config: any): {
  isValid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check required fields
  if (!config.categoricalFeatures || !Array.isArray(config.categoricalFeatures)) {
    errors.push('categoricalFeatures must be an array');
  }
  
  if (!config.numericFeatures || !Array.isArray(config.numericFeatures)) {
    errors.push('numericFeatures must be an array');
  }
  
  if (!config.target || typeof config.target !== 'string') {
    errors.push('target must be a string');
  }
  
  if (!config.taskType || !['classification', 'regression'].includes(config.taskType)) {
    errors.push('taskType must be either "classification" or "regression"');
  }

  // Check for warnings
  if (config.categoricalFeatures.length === 0 && config.numericFeatures.length === 0) {
    warnings.push('No features specified - model will not be trainable');
  }
  
  if (config.categoricalFeatures.length > 100) {
    warnings.push('Large number of categorical features may impact performance');
  }
  
  if (config.taskType === 'regression' && config.xgbParams?.objective?.startsWith('binary:')) {
    warnings.push('Binary objective specified for regression task - this may not work as expected');
  }
  
  if (config.taskType === 'classification' && config.xgbParams?.objective?.startsWith('reg:')) {
    warnings.push('Regression objective specified for classification task - this may not work as expected');
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validates the current environment for ML operations
 * 
 * @returns Environment validation result
 */
export function validateEnvironment(): {
  isValid: boolean;
  issues: string[];
  environment: 'node' | 'browser' | 'unknown';
  features: {
    fileSystem: boolean;
    performance: boolean;
    crypto: boolean;
  };
} {
  const issues: string[] = [];
  let environment: 'node' | 'browser' | 'unknown' = 'unknown';
  
  // Detect environment
  if (typeof window !== 'undefined') {
    environment = 'browser';
  } else if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    environment = 'node';
  }
  
  // Check features
  const features = {
    fileSystem: false,
    performance: false,
    crypto: false,
  };
  
  // File system check
  try {
    if (typeof require !== 'undefined') {
      require('fs');
      features.fileSystem = true;
    }
  } catch (e) {
    if (environment === 'node') {
      issues.push('File system access not available');
    }
  }
  
  // Performance API check
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    features.performance = true;
  } else {
    issues.push('High-resolution performance timing not available');
  }
  
  // Crypto check
  if (typeof crypto !== 'undefined' || (typeof require !== 'undefined' && require('crypto'))) {
    features.crypto = true;
  } else {
    issues.push('Crypto API not available');
  }
  
  return {
    isValid: issues.length === 0,
    issues,
    environment,
    features,
  };
}

/**
 * Gets performance metrics for the current environment
 * 
 * @returns Performance metrics object
 */
export function getPerformanceMetrics(): {
  memoryUsage?: NodeJS.MemoryUsage;
  timing: {
    highResolution: boolean;
    precision: number;
  };
} {
  const metrics: any = {
    timing: {
      highResolution: typeof performance !== 'undefined' && typeof performance.now === 'function',
      precision: typeof performance !== 'undefined' ? 0.001 : 1, // milliseconds
    },
  };
  
  // Node.js specific metrics
  if (typeof process !== 'undefined' && process.memoryUsage) {
    metrics.memoryUsage = process.memoryUsage();
  }
  
  return metrics;
}

/**
 * Creates a test dataset for development and testing
 * 
 * @param size - Number of samples to generate
 * @param type - Type of dataset ('binary', 'multiclass', or 'regression')
 * @returns Array of test data samples
 */
export function createTestDataset(size: number, type: 'binary' | 'multiclass' | 'regression' = 'binary'): any[] {
  const samples: any[] = [];
  const categories = ['A', 'B', 'C', 'D'];
  const brands = ['Brand1', 'Brand2', 'Brand3'];
  const colors = ['red', 'blue', 'green', 'yellow'];
  
  for (let i = 0; i < size; i++) {
    const category = categories[Math.floor(Math.random() * categories.length)];
    const brand = brands[Math.floor(Math.random() * brands.length)];
    const color = colors[Math.floor(Math.random() * colors.length)];
    const price = Math.random() * 1000 + 100;
    const rating = Math.random() * 5;
    const sales = Math.floor(Math.random() * 10000);
    
    let target;
    if (type === 'binary') {
      target = price > 500 ? 'expensive' : 'cheap';
    } else if (type === 'multiclass') {
      if (price > 750) target = 'premium';
      else if (price > 400) target = 'mid';
      else target = 'budget';
    } else {
      // Regression: predict price based on features
      target = price + (rating * 50) + (sales * 0.01) + (Math.random() * 100 - 50);
    }
    
    samples.push({
      category,
      brand,
      color,
      price,
      rating,
      sales,
      target,
    });
  }
  
  return samples;
} 