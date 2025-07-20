/**
 * @fileoverview Core type definitions for the XGBoost Decision Tree Toolkit
 * This file contains all the TypeScript interfaces and types used throughout the project
 */

/**
 * Feature specification containing unique count and bucket count for categorical features
 */
export interface FeatureSpec {
  /** The unique count of values for this categorical feature */
  k: number;
  /** The bucket count for hashing (calculated as k² for k≤1000, otherwise 20×k) */
  bucketCount: number;
}

/**
 * Mapping of feature names to their specifications
 */
export interface FeatureSpecMap {
  [featureName: string]: FeatureSpec;
}

/**
 * Configuration parameters for XGBoost training
 */
export interface XgbParams {
  /** Training objective (e.g., 'binary:logistic', 'multi:softprob', 'reg:squarederror') */
  objective?: string;
  /** Maximum depth of trees */
  max_depth?: number;
  /** Learning rate (eta) */
  eta?: number;
  /** Number of classes for multi-class classification */
  num_class?: number;
  /** Number of boosting rounds */
  num_boost_round?: number;
  /** Minimum child weight */
  min_child_weight?: number;
  /** Gamma (minimum loss reduction) */
  gamma?: number;
  /** Subsample ratio of training instances */
  subsample?: number;
  /** Subsample ratio of features */
  colsample_bytree?: number;
  /** L1 regularization term on weights */
  alpha?: number;
  /** L2 regularization term on weights */
  lambda?: number;
  /** Random seed for reproducibility */
  seed?: number;
  /** Number of parallel threads */
  nthread?: number;
  /** Verbosity level */
  verbosity?: number;
  /** Booster type: 'gbtree', 'gblinear', or 'dart' */
  booster?: string;
  /** Evaluation metric (e.g., 'rmse', 'error', 'auc') */
  eval_metric?: string;
}

/**
 * Task type for training - either classification or regression
 */
export type TaskType = 'classification' | 'regression';

/**
 * Configuration for model training - extended to support regression
 */
export interface TrainingConfig {
  /** Whether to use cross-validation during training */
  useCrossValidation?: boolean;
  /** Number of folds for cross-validation */
  nFolds?: number;
  /** Test set ratio for train-test split */
  testRatio?: number;
  /** Whether to shuffle data before splitting */
  shuffle?: boolean;
  /** Random seed for data shuffling */
  seed?: number;
  /** Task type - classification or regression */
  taskType?: TaskType;
}

/**
 * Raw data record interface - flexible object with any properties
 */
export interface RawDataRecord {
  [key: string]: string | number | boolean | null | undefined;
}

/**
 * Dataset interface for structured data
 */
export interface Dataset {
  /** Array of raw data records */
  data: RawDataRecord[];
  /** Names of categorical features */
  categoricalFeatures: string[];
  /** Names of numerical features */
  numericFeatures: string[];
  /** Name of the target variable */
  target: string;
}

/**
 * Training and validation split
 */
export interface DataSplit {
  /** Training dataset */
  train: Dataset;
  /** Validation/test dataset */
  test: Dataset;
}

/**
 * Model evaluation metrics - extended to support regression
 */
export interface ModelMetrics {
  /** Overall accuracy (0-1) - classification only */
  accuracy?: number;
  /** ROC AUC score (0-1) - classification only */
  rocAuc?: number;
  /** Confusion matrix [actual][predicted] - classification only */
  confusionMatrix?: number[][];
  /** Precision per class - classification only */
  precision?: number[];
  /** Recall per class - classification only */
  recall?: number[];
  /** F1-score per class - classification only */
  f1Score?: number[];
  /** Cross-validation scores (if performed) */
  crossValidationScores?: number[];
  /** Task type this metric applies to */
  taskType: TaskType;
  
  // Regression-specific metrics
  /** Mean Squared Error - regression only */
  mse?: number;
  /** Root Mean Squared Error - regression only */
  rmse?: number;
  /** Mean Absolute Error - regression only */
  mae?: number;
  /** R-squared (coefficient of determination) - regression only */
  r2?: number;
  /** Mean Absolute Percentage Error - regression only */
  mape?: number;
  /** Median Absolute Error - regression only */
  medianAe?: number;
}

/**
 * Encoder metadata for model persistence
 */
export interface EncoderMetadata {
  /** Feature name */
  featureName: string;
  /** Bucket count for hashing */
  bucketCount: number;
  /** Hash seed for reproducibility */
  hashSeed: number;
  /** Unique value count */
  uniqueCount: number;
}

/**
 * Model metadata for persistence and loading - extended for regression
 */
export interface ModelMetadata {
  /** Model version */
  version: string;
  /** Timestamp of model creation */
  createdAt: string;
  /** Categorical feature names */
  categoricalFeatures: string[];
  /** Numerical feature names */
  numericFeatures: string[];
  /** Target variable name */
  target: string;
  /** Task type - classification or regression */
  taskType: TaskType;
  /** Unique class labels - classification only */
  classes?: string[];
  /** Number of classes - classification only */
  numClasses?: number;
  /** XGBoost parameters used for training */
  xgbParams: XgbParams;
  /** Encoder metadata for each categorical feature */
  encoders: EncoderMetadata[];
  /** Training metrics */
  metrics: ModelMetrics;
  /** Whether this is a multi-class model using One-vs-Rest - classification only */
  isOneVsRest?: boolean;
  /** Target statistics for regression models */
  targetStats?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  /** Target normalization parameters for regression models */
  targetNormalization?: {
    min: number;
    max: number;
  };
}

/**
 * Prediction result interface - extended for regression
 */
export interface PredictionResult {
  /** Task type this prediction applies to */
  taskType: TaskType;
  
  // Classification-specific fields
  /** Predicted class index - classification only */
  classIndex?: number;
  /** Predicted class label - classification only */
  classLabel?: string;
  /** Prediction probability/confidence - classification only */
  probability?: number;
  /** All class probabilities (for multi-class) - classification only */
  probabilities?: number[];
  
  // Regression-specific fields
  /** Predicted numeric value - regression only */
  value?: number;
  /** Prediction confidence interval - regression only */
  confidenceInterval?: [number, number];
}

/**
 * Batch prediction results - extended for regression
 */
export interface BatchPredictionResult {
  /** Task type this prediction applies to */
  taskType: TaskType;
  /** Array of prediction results */
  predictions: PredictionResult[];
  /** Overall prediction confidence - classification only */
  averageConfidence?: number;
  /** Overall prediction statistics - regression only */
  predictionStats?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
}

/**
 * Logging levels
 */
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

/**
 * Logger interface
 */
export interface Logger {
  debug(message: string, meta?: any): void;
  info(message: string, meta?: any): void;
  warn(message: string, meta?: any): void;
  error(message: string, meta?: any): void;
}

/**
 * Configuration for data loading
 */
export interface DataLoadConfig {
  /** Delimiter for CSV files */
  delimiter?: string;
  /** Whether CSV has header row */
  hasHeader?: boolean;
  /** Encoding for file reading */
  encoding?: string;
  /** Maximum number of rows to read */
  maxRows?: number;
  /** Skip first N rows */
  skipRows?: number;
}

/**
 * Hash encoder configuration
 */
export interface HashEncoderConfig {
  /** Number of hash buckets */
  bucketCount: number;
  /** Hash seed for reproducibility */
  hashSeed: number;
  /** Feature name */
  featureName: string;
}

/**
 * Feature analysis result
 */
export interface FeatureAnalysisResult {
  /** Feature specifications for categorical features */
  categoricalSpecs: FeatureSpecMap;
  /** Summary statistics for numerical features */
  numericStats: {
    [featureName: string]: {
      min: number;
      max: number;
      mean: number;
      std: number;
      nullCount: number;
    };
  };
  /** Overall dataset statistics */
  datasetStats: {
    totalRows: number;
    totalFeatures: number;
    categoricalFeatures: number;
    numericFeatures: number;
    targetClasses: string[];
    nullValues: number;
  };
}

/**
 * Cross-validation configuration
 */
export interface CrossValidationConfig {
  /** Number of folds */
  nFolds: number;
  /** Whether to shuffle data */
  shuffle: boolean;
  /** Random seed */
  seed: number;
  /** Stratified sampling for classification */
  stratified: boolean;
}

/**
 * Model training progress callback
 */
export interface TrainingProgressCallback {
  (progress: {
    iteration: number;
    totalIterations: number;
    currentMetric: number;
    bestMetric: number;
    elapsedTime: number;
  }): void;
}

/**
 * Environment detection result
 */
export interface Environment {
  isNode: boolean;
  isBrowser: boolean;
  nodeVersion?: string;
  browserType?: string;
}

/**
 * File operation result
 */
export interface FileOperationResult {
  /** Whether operation was successful */
  success: boolean;
  /** Error message if operation failed */
  error?: string;
  /** File path that was operated on */
  filePath: string;
  /** Size of file in bytes */
  size?: number;
}

/**
 * Model validation result
 */
export interface ModelValidationResult {
  /** Whether model is valid */
  isValid: boolean;
  /** Validation errors */
  errors: string[];
  /** Validation warnings */
  warnings: string[];
  /** Model compatibility version */
  compatibilityVersion: string;
}

/**
 * Regression-specific metrics interface
 */
export interface RegressionMetrics {
  /** Mean Squared Error */
  mse: number;
  /** Root Mean Squared Error */
  rmse: number;
  /** Mean Absolute Error */
  mae: number;
  /** R-squared (coefficient of determination) */
  r2: number;
  /** Mean Absolute Percentage Error */
  mape: number;
  /** Median Absolute Error */
  medianAe: number;
  /** Number of samples */
  n: number;
}

/**
 * Regression training configuration
 */
export interface RegressionTrainingConfig extends TrainingConfig {
  /** Task type - must be 'regression' */
  taskType: 'regression';
  /** XGBoost objective for regression */
  objective?: 'reg:squarederror' | 'reg:logistic' | 'reg:pseudohubererror' | 'reg:gamma' | 'reg:tweedie';
  /** Evaluation metric for regression */
  evalMetric?: 'rmse' | 'mae' | 'mape' | 'mphe' | 'rmsle';
}

/**
 * Classification training configuration
 */
export interface ClassificationTrainingConfig extends TrainingConfig {
  /** Task type - must be 'classification' */
  taskType: 'classification';
  /** XGBoost objective for classification */
  objective?: 'binary:logistic' | 'multi:softprob' | 'multi:softmax';
  /** Evaluation metric for classification */
  evalMetric?: 'logloss' | 'mlogloss' | 'auc' | 'aucpr' | 'error';
}

/**
 * Export all types for easy importing
 */
export * from './index';

/**
 * Tree node structure for visualization
 */
export interface TreeNode {
  /** Feature index for splits (-1 for leaf nodes) */
  featureIndex: number | null;
  /** Split threshold value */
  threshold: number | null;
  /** Left child node */
  left: TreeNode | null;
  /** Right child node */
  right: TreeNode | null;
  /** Leaf value (for leaf nodes) */
  value: number | null;
  /** Whether this is a leaf node */
  isLeaf: boolean;
  /** Node depth in the tree */
  depth?: number;
  /** Node ID for reference */
  nodeId?: string;
}

/**
 * Tree visualization options
 */
export interface TreeVisualizationOptions {
  /** Include feature names instead of indices */
  includeFeatureNames?: boolean;
  /** Maximum depth to display */
  maxDepth?: number;
  /** Include leaf values */
  includeValues?: boolean;
  /** Precision for floating point numbers */
  precision?: number;
  /** Tree index to visualize (for multi-tree models) */
  treeIndex?: number;
  /** Class name to visualize (for One-vs-Rest models) */
  className?: string;
}

/**
 * Tree visualization formats
 */
export type VisualizationFormat = 'text' | 'json' | 'html' | 'svg' | 'uml';

/**
 * Tree visualization result
 */
export interface TreeVisualizationResult {
  /** Visualization format */
  format: VisualizationFormat;
  /** Generated content */
  content: string;
  /** Tree metadata */
  metadata: {
    treeIndex: number;
    maxDepth: number;
    nodeCount: number;
    leafCount: number;
  };
} 