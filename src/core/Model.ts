/**
 * @fileoverview Model implementation for loading and inference
 * 
 * The Model class handles loading trained models from disk and performing
 * inference on new data with proper feature transformation and encoding.
 */

import { XGBoost } from '@fractal-solutions/xgboost-js';
import { HashEncoder } from './HashEncoder';
import { TreeVisualizer } from '../utils/TreeVisualizer';
import {
  RawDataRecord,
  ModelMetadata,
  PredictionResult,
  BatchPredictionResult,
  Logger,
  TaskType,
  TreeVisualizationOptions,
  VisualizationFormat,
  TreeVisualizationResult,
} from '../types';

/**
 * Model class for loading and inference
 * 
 * Features:
 * - Loading models from saved artifacts
 * - Feature transformation using saved encoders
 * - Single and batch prediction for classification and regression
 * - Probability distribution output for classification
 * - Numeric value output for regression
 * - Metadata access and validation
 * 
 * @example
 * ```typescript
 * // Classification model
 * const classificationModel = await Model.load('./models/classification_model');
 * const prediction = classificationModel.predict({ 
 *   brand: 'Apple', 
 *   color: 'red', 
 *   price: 99.99 
 * });
 * console.log('Predicted class:', prediction); // 0 or 1
 * 
 * // Regression model
 * const regressionModel = await Model.load('./models/regression_model');
 * const value = regressionModel.predictValue({ 
 *   brand: 'Apple', 
 *   color: 'red', 
 *   area: 1200 
 * });
 * console.log('Predicted value:', value); // 450000.5
 * ```
 */
export class Model {
  private metadata: ModelMetadata;
  private encoders: Map<string, HashEncoder> = new Map();
  private model: XGBoost | undefined;
  private oneVsRestModels: Map<string, XGBoost> = new Map();
  private logger: Logger | undefined;

  /**
   * Creates a new Model instance (use Model.load() to load from disk)
   * 
   * @private
   */
  private constructor(metadata: ModelMetadata, logger?: Logger) {
    this.metadata = metadata;
    this.logger = logger;
    
    // Initialize encoders from metadata
    for (const encoderMeta of metadata.encoders) {
      const encoder = new HashEncoder(
        encoderMeta.bucketCount,
        encoderMeta.hashSeed,
        encoderMeta.featureName
      );
      this.encoders.set(encoderMeta.featureName, encoder);
    }
  }

  /**
   * Loads a model from a directory
   * 
   * @param directory - Directory containing the saved model
   * @param logger - Optional logger for debugging
   * @returns Promise resolving to loaded Model instance
   * 
   * @throws Error if model files are missing or invalid
   * 
   * @example
   * ```typescript
   * const model = await Model.load('./models/sales_model');
   * console.log('Model loaded:', model.getMetadata().version);
   * ```
   */
  public static async load(directory: string, logger?: Logger): Promise<Model> {
    logger?.info('Loading model from directory', { directory });
    
    const fs = require('fs');
    
    // Check if directory exists
    if (!fs.existsSync(directory)) {
      throw new Error(`Model directory not found: ${directory}`);
    }
    
    // Load metadata
    const metadataPath = `${directory}/metadata.json`;
    if (!fs.existsSync(metadataPath)) {
      throw new Error(`Model metadata not found: ${metadataPath}`);
    }
    
    const metadataContent = fs.readFileSync(metadataPath, 'utf8');
    const metadata: ModelMetadata = JSON.parse(metadataContent);
    
    // Create model instance
    const model = new Model(metadata, logger);
    
    // Load XGBoost model(s)
    if (metadata.isOneVsRest) {
      // Load One-vs-Rest models
      const modelsDir = `${directory}/models`;
      if (!fs.existsSync(modelsDir)) {
        throw new Error(`One-vs-Rest models directory not found: ${modelsDir}`);
      }
      
      for (const className of metadata.classes || []) {
        const modelPath = `${modelsDir}/${className}.json`;
        if (fs.existsSync(modelPath)) {
          const modelContent = fs.readFileSync(modelPath, 'utf8');
          const modelJSON = JSON.parse(modelContent);
          const xgbModel = XGBoost.fromJSON(modelJSON);
          model.oneVsRestModels.set(className, xgbModel);
        }
      }
      
      if (model.oneVsRestModels.size === 0) {
        throw new Error('No One-vs-Rest models found');
      }
    } else {
      // Load single model
      const modelPath = `${directory}/model.json`;
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Model file not found: ${modelPath}`);
      }
      
      const modelContent = fs.readFileSync(modelPath, 'utf8');
      const modelJSON = JSON.parse(modelContent);
      model.model = XGBoost.fromJSON(modelJSON);
    }
    
    logger?.info('Model loaded successfully', {
      version: metadata.version,
      classes: metadata.classes?.length,
      isOneVsRest: metadata.isOneVsRest,
    });
    
    return model;
  }

  /**
   * Predicts class for a single data record (classification only)
   * 
   * @param record - Raw data record to predict
   * @returns Predicted class index
   * 
   * @throws Error if model is not a classification model
   * 
   * @example
   * ```typescript
   * const prediction = model.predict({ 
   *   brand: 'Apple', 
   *   color: 'red', 
   *   price: 99.99 
   * });
   * console.log('Predicted class:', prediction); // 0 or 1
   * ```
   */
  public predict(record: RawDataRecord): number {
    if (this.metadata.taskType !== 'classification') {
      throw new Error('predict() method is only available for classification models. Use predictValue() for regression.');
    }
    
    const result = this.predictWithProbabilities(record);
    return result.classIndex!;
  }

  /**
   * Predicts numeric value for a single data record (regression only)
   * 
   * @param record - Raw data record to predict
   * @returns Predicted numeric value
   * 
   * @throws Error if model is not a regression model
   * 
   * @example
   * ```typescript
   * const value = model.predictValue({ 
   *   brand: 'Apple', 
   *   color: 'red', 
   *   area: 1200 
   * });
   * console.log('Predicted value:', value); // 450000.5
   * ```
   */
  public predictValue(record: RawDataRecord): number {
    if (this.metadata.taskType !== 'regression') {
      throw new Error('predictValue() method is only available for regression models. Use predict() for classification.');
    }
    
    const result = this.predictWithProbabilities(record);
    return result.value!;
  }

  /**
   * Predicts classes for multiple data records (classification only)
   * 
   * @param records - Array of raw data records to predict
   * @returns Array of predicted class indices
   * 
   * @throws Error if model is not a classification model
   * 
   * @example
   * ```typescript
   * const predictions = model.predictBatch([
   *   { brand: 'Apple', color: 'red', price: 99.99 },
   *   { brand: 'Samsung', color: 'blue', price: 79.99 },
   * ]);
   * console.log('Predictions:', predictions); // [0, 1]
   * ```
   */
  public predictBatch(records: RawDataRecord[]): number[] {
    if (this.metadata.taskType !== 'classification') {
      throw new Error('predictBatch() method is only available for classification models. Use predictValueBatch() for regression.');
    }
    
    return records.map(record => this.predict(record));
  }

  /**
   * Predicts numeric values for multiple data records (regression only)
   * 
   * @param records - Array of raw data records to predict
   * @returns Array of predicted numeric values
   * 
   * @throws Error if model is not a regression model
   * 
   * @example
   * ```typescript
   * const values = model.predictValueBatch([
   *   { brand: 'Apple', color: 'red', area: 1200 },
   *   { brand: 'Samsung', color: 'blue', area: 800 },
   * ]);
   * console.log('Predicted values:', values); // [450000.5, 320000.8]
   * ```
   */
  public predictValueBatch(records: RawDataRecord[]): number[] {
    if (this.metadata.taskType !== 'regression') {
      throw new Error('predictValueBatch() method is only available for regression models. Use predictBatch() for classification.');
    }
    
    return records.map(record => this.predictValue(record));
  }

  /**
   * Predicts class probabilities for a single data record (classification only)
   * 
   * @param record - Raw data record to predict
   * @returns Array of probabilities for each class
   * 
   * @throws Error if model is not a classification model
   * 
   * @example
   * ```typescript
   * const probabilities = model.predictProbabilities({ 
   *   brand: 'Apple', 
   *   color: 'red', 
   *   price: 99.99 
   * });
   * console.log('Class probabilities:', probabilities); // [0.2, 0.8]
   * ```
   */
  public predictProbabilities(record: RawDataRecord): number[] {
    if (this.metadata.taskType !== 'classification') {
      throw new Error('predictProbabilities() method is only available for classification models.');
    }
    
    const result = this.predictWithProbabilities(record);
    return result.probabilities!;
  }

  /**
   * Predicts class with full probability information (classification only)
   * 
   * @param record - Raw data record to predict
   * @returns Complete prediction result with probabilities
   * 
   * @throws Error if model is not a classification model
   * 
   * @example
   * ```typescript
   * const result = model.predictWithProbabilities({ 
   *   brand: 'Apple', 
   *   color: 'red', 
   *   price: 99.99 
   * });
   * console.log('Class:', result.classLabel);
   * console.log('Confidence:', result.probability);
   * ```
   */
  public predictWithProbabilities(record: RawDataRecord): PredictionResult {
    // Transform features
    const features = this.transform(record);
    
    if (this.metadata.taskType === 'classification') {
      return this.predictClassification(features);
    } else {
      return this.predictRegression(features);
    }
  }

  /**
   * Predicts with full information for multiple records
   * 
   * @param records - Array of raw data records to predict
   * @returns Batch prediction result with statistics
   * 
   * @example
   * ```typescript
   * const result = model.predictBatchWithProbabilities([
   *   { brand: 'Apple', color: 'red', price: 99.99 },
   *   { brand: 'Samsung', color: 'blue', price: 79.99 },
   * ]);
   * console.log('Predictions:', result.predictions);
   * console.log('Average confidence:', result.averageConfidence);
   * ```
   */
  public predictBatchWithProbabilities(records: RawDataRecord[]): BatchPredictionResult {
    const predictions = records.map(record => this.predictWithProbabilities(record));
    
    const result: BatchPredictionResult = {
      taskType: this.metadata.taskType,
      predictions,
    };
    
    if (this.metadata.taskType === 'classification') {
      // Calculate average confidence for classification
      const confidences = predictions.map(p => p.probability!);
      result.averageConfidence = confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    } else {
      // Calculate prediction statistics for regression
      const values = predictions.map(p => p.value!);
      result.predictionStats = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean: values.reduce((sum, val) => sum + val, 0) / values.length,
        std: Math.sqrt(values.reduce((sum, val) => sum + Math.pow(val - result.predictionStats!.mean, 2), 0) / values.length),
      };
    }
    
    return result;
  }

  /**
   * Transforms a raw data record to feature vector
   * 
   * @param record - Raw data record to transform
   * @returns Feature vector as number array
   * 
   * @example
   * ```typescript
   * const features = model.transform({ 
   *   brand: 'Apple', 
   *   color: 'red', 
   *   price: 99.99 
   * });
   * console.log('Features:', features); // [0, 1, 0, 0, 99.99]
   * ```
   */
  public transform(record: RawDataRecord): number[] {
    const features: number[] = [];
    
    // Add categorical features (encoded)
    for (const featureName of this.metadata.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        const encoded = encoder.encode(record[featureName]);
        features.push(...Array.from(encoded));
      } else {
        this.logger?.warn(`Encoder not found for feature: ${featureName}`);
      }
    }
    
    // Add numerical features
    for (const featureName of this.metadata.numericFeatures) {
      const value = record[featureName];
      if (typeof value === 'number') {
        features.push(value);
      } else if (typeof value === 'string' && !isNaN(Number(value))) {
        features.push(Number(value));
      } else {
        // Handle missing or invalid numeric values
        features.push(0);
        this.logger?.warn(`Invalid numeric value for feature ${featureName}: ${value}`);
      }
    }
    
    return features;
  }

  /**
   * Transforms multiple raw data records to feature vectors
   * 
   * @param records - Array of raw data records to transform
   * @returns Array of feature vectors
   * 
   * @example
   * ```typescript
   * const features = model.transformBatch([
   *   { brand: 'Apple', color: 'red', price: 99.99 },
   *   { brand: 'Samsung', color: 'blue', price: 79.99 },
   * ]);
   * console.log('Features shape:', features.length, 'x', features[0].length);
   * ```
   */
  public transformBatch(records: RawDataRecord[]): number[][] {
    return records.map(record => this.transform(record));
  }

  /**
   * Gets model metadata
   * 
   * @returns Model metadata
   * 
   * @example
   * ```typescript
   * const metadata = model.getMetadata();
   * console.log('Model version:', metadata.version);
   * console.log('Classes:', metadata.classes);
   * console.log('Features:', metadata.categoricalFeatures, metadata.numericFeatures);
   * ```
   */
  public getMetadata(): ModelMetadata {
    return { ...this.metadata };
  }

  /**
   * Gets the list of feature names in the order they are used
   * 
   * @returns Array of feature names
   */
  public getFeatureNames(): string[] {
    return [...this.metadata.categoricalFeatures, ...this.metadata.numericFeatures];
  }

  /**
   * Returns the classes for this model (classification only)
   * 
   * @returns Array of class names
   * 
   * @throws Error if model is not a classification model
   */
  public getClasses(): string[] {
    if (this.metadata.taskType !== 'classification' || !this.metadata.classes) {
      throw new Error('getClasses() is only available for classification models');
    }
    return this.metadata.classes;
  }

  /**
   * Gets the number of features in the model
   * 
   * @returns Total number of features
   */
  public getFeatureCount(): number {
    let count = 0;
    
    // Count categorical features (encoded dimensions)
    for (const featureName of this.metadata.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        count += encoder.getDimension();
      }
    }
    
    // Count numerical features
    count += this.metadata.numericFeatures.length;
    
    return count;
  }

  /**
   * Validates that a data record has all required features
   * 
   * @param record - Raw data record to validate
   * @returns Validation result
   * 
   * @example
   * ```typescript
   * const validation = model.validateRecord({ brand: 'Apple', color: 'red' });
   * if (!validation.isValid) {
   *   console.error('Missing features:', validation.missingFeatures);
   * }
   * ```
   */
  public validateRecord(record: RawDataRecord): {
    isValid: boolean;
    missingFeatures: string[];
    extraFeatures: string[];
  } {
    const requiredFeatures = this.getFeatureNames();
    const providedFeatures = Object.keys(record);
    
    const missingFeatures = requiredFeatures.filter(feature => 
      !(feature in record) || record[feature] === undefined
    );
    
    const extraFeatures = providedFeatures.filter(feature => 
      !requiredFeatures.includes(feature)
    );
    
    return {
      isValid: missingFeatures.length === 0,
      missingFeatures,
      extraFeatures,
    };
  }

  /**
   * Gets model performance metrics from training
   * 
   * @returns Training metrics
   */
  public getTrainingMetrics(): any {
    return this.metadata.metrics;
  }

  /**
   * Returns whether this is a One-vs-Rest model
   * 
   * @returns True if One-vs-Rest model
   */
  public isOneVsRest(): boolean {
    return this.metadata.isOneVsRest === true;
  }

  /**
   * Gets the model creation timestamp
   * 
   * @returns ISO timestamp string
   */
  public getCreatedAt(): string {
    return this.metadata.createdAt;
  }

  /**
   * Gets the model version
   * 
   * @returns Version string
   */
  public getVersion(): string {
    return this.metadata.version;
  }

  /**
   * Returns string representation of the model
   * 
   * @returns String representation
   */
  public toString(): string {
    if (this.metadata.taskType === 'classification') {
      const numClasses = this.metadata.classes?.length || 0;
      return `Model(version=${this.metadata.version}, classes=${numClasses}, features=${this.getFeatureCount()}, oneVsRest=${this.metadata.isOneVsRest === true})`;
    } else {
      return `Model(version=${this.metadata.version}, taskType=regression, features=${this.getFeatureCount()})`;
    }
  }

  /**
   * Returns the task type of the model
   * 
   * @returns Task type ('classification' or 'regression')
   */
  public getTaskType(): TaskType {
    return this.metadata.taskType;
  }

  /**
   * Returns whether this is a classification model
   * 
   * @returns True if classification model
   */
  public isClassification(): boolean {
    return this.metadata.taskType === 'classification';
  }

  /**
   * Returns whether this is a regression model
   * 
   * @returns True if regression model
   */
  public isRegression(): boolean {
    return this.metadata.taskType === 'regression';
  }

  /**
   * Returns target statistics for regression models
   * 
   * @returns Target statistics or undefined for classification models
   */
  public getTargetStats(): { min: number; max: number; mean: number; std: number; } | undefined {
    return this.metadata.targetStats;
  }

  /**
   * Visualizes a decision tree from the loaded model
   * 
   * @param format - Visualization format ('text', 'json', 'html', 'svg')
   * @param options - Visualization options
   * @returns Tree visualization result
   * 
   * @example
   * ```typescript
   * const model = await Model.load('./models/my_model');
   * 
   * // Text visualization
   * const textViz = model.visualizeTree('text', { treeIndex: 0, includeFeatureNames: true });
   * console.log(textViz.content);
   * 
   * // HTML visualization
   * const htmlViz = model.visualizeTree('html', { maxDepth: 3, precision: 2 });
   * ```
   */
  public visualizeTree(
    format: VisualizationFormat,
    options: TreeVisualizationOptions = {}
  ): TreeVisualizationResult {
    if (!this.model && this.oneVsRestModels.size === 0) {
      throw new Error('No model loaded');
    }

    // Create feature names array for the visualizer
    const featureNames: string[] = [];
    
    // Add categorical features (expanded for hash encoding)
    for (const featureName of this.metadata.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        const dimension = encoder.getDimension();
        for (let i = 0; i < dimension; i++) {
          featureNames.push(`${featureName}_${i}`);
        }
      }
    }
    
    // Add numeric features
    featureNames.push(...this.metadata.numericFeatures);

    const visualizer = new TreeVisualizer(featureNames, this.logger);

    if (this.metadata.isOneVsRest) {
      // For One-vs-Rest, visualize a specific class model
      if (!this.metadata.classes || this.metadata.classes.length === 0) {
        throw new Error('No classes available for One-vs-Rest model');
      }
      
      const className = options.className || this.metadata.classes[0]!;
      const classModel = this.oneVsRestModels.get(className);
      
      if (!classModel) {
        throw new Error(`One-vs-Rest model not found for class: ${className}`);
      }
      
      return visualizer.visualizeTree(classModel, format, options);
    } else {
      // Single model visualization
      return visualizer.visualizeTree(this.model!, format, options);
    }
  }

  /**
   * Saves tree visualization to a file
   * 
   * @param filePath - Path to save the visualization
   * @param format - Visualization format
   * @param options - Visualization options
   * 
   * @example
   * ```typescript
   * const model = await Model.load('./models/my_model');
   * 
   * // Save HTML visualization
   * await model.saveTreeVisualization('./tree.html', 'html', { includeFeatureNames: true });
   * 
   * // Save SVG visualization
   * await model.saveTreeVisualization('./tree.svg', 'svg', { maxDepth: 5 });
   * ```
   */
  public async saveTreeVisualization(
    filePath: string,
    format: VisualizationFormat,
    options: TreeVisualizationOptions = {}
  ): Promise<void> {
    const result = this.visualizeTree(format, options);
    
    const fs = require('fs');
    fs.writeFileSync(filePath, result.content, 'utf8');
    
    this.logger?.info('Tree visualization saved', { filePath, format, metadata: result.metadata });
  }

  /**
   * Gets tree statistics from the loaded model
   * 
   * @returns Tree statistics for all trees in the model
   * 
   * @example
   * ```typescript
   * const model = await Model.load('./models/my_model');
   * const stats = model.getTreeStatistics();
   * console.log(`Model has ${stats.treeCount} trees`);
   * stats.trees.forEach(tree => {
   *   console.log(`Tree ${tree.index}: ${tree.nodeCount} nodes, depth ${tree.maxDepth}`);
   * });
   * ```
   */
  public getTreeStatistics(): {
    treeCount: number;
    trees: Array<{
      index: number;
      maxDepth: number;
      nodeCount: number;
      leafCount: number;
    }>;
  } {
    if (!this.model && this.oneVsRestModels.size === 0) {
      throw new Error('No model loaded');
    }

    // Create feature names array for the visualizer
    const featureNames: string[] = [];
    
    // Add categorical features (expanded for hash encoding)
    for (const featureName of this.metadata.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        const dimension = encoder.getDimension();
        for (let i = 0; i < dimension; i++) {
          featureNames.push(`${featureName}_${i}`);
        }
      }
    }
    
    // Add numeric features
    featureNames.push(...this.metadata.numericFeatures);

    const visualizer = new TreeVisualizer(featureNames, this.logger);

    if (this.metadata.isOneVsRest) {
      // For One-vs-Rest, return statistics for first class model
      if (!this.metadata.classes || this.metadata.classes.length === 0) {
        throw new Error('No classes available for One-vs-Rest model');
      }
      
      const className = this.metadata.classes[0]!;
      const classModel = this.oneVsRestModels.get(className);
      
      if (!classModel) {
        throw new Error(`One-vs-Rest model not found for class: ${className}`);
      }
      
      return visualizer.getTreeStatistics(classModel);
    } else {
      // Single model statistics
      return visualizer.getTreeStatistics(this.model!);
    }
  }

  /**
   * Performs classification prediction on transformed features
   * 
   * @private
   */
  private predictClassification(features: number[]): PredictionResult {
    if (this.metadata.isOneVsRest === true) {
      // One-vs-Rest prediction
      const classScores = new Map<string, number>();
      
      for (const [className, model] of this.oneVsRestModels.entries()) {
        const prob = model.predictSingle(features);
        classScores.set(className, prob);
      }
      
      // Find the class with highest score
      let bestClass = '';
      let bestScore = -Infinity;
      
      for (const [className, score] of classScores.entries()) {
        if (score > bestScore) {
          bestScore = score;
          bestClass = className;
        }
      }
      
      const classes = this.metadata.classes || [];
      const classIndex = classes.indexOf(bestClass);
      const probabilities = classes.map(cls => classScores.get(cls) || 0);
      
      return {
        taskType: 'classification',
        classIndex,
        classLabel: bestClass || 'unknown',
        probability: bestScore,
        probabilities,
      };
    } else {
      // Single model prediction
      if (!this.model) {
        throw new Error('Model not loaded');
      }
      
      const prob = this.model.predictSingle(features);
      const classes = this.metadata.classes || [];
      const numClasses = this.metadata.numClasses || 2;
      
      if (numClasses > 2) {
        // Multi-class (though XGBoost might handle this differently)
        const classIndex = prob > 0.5 ? 1 : 0;
        return {
          taskType: 'classification',
          classIndex,
          classLabel: classes[classIndex] || 'unknown',
          probability: prob,
          probabilities: [1 - prob, prob],
        };
      } else {
        // Binary classification
        const classIndex = prob > 0.5 ? 1 : 0;
        return {
          taskType: 'classification',
          classIndex,
          classLabel: classes[classIndex] || 'unknown',
          probability: prob,
          probabilities: [1 - prob, prob],
        };
      }
    }
  }

  /**
   * Performs regression prediction on transformed features
   * 
   * @private
   */
  private predictRegression(features: number[]): PredictionResult {
    if (!this.model) {
      throw new Error('Model not loaded');
    }
    
    const normalizedValue = this.model.predictSingle(features);
    
    // Denormalize the prediction if normalization parameters are available
    let denormalizedValue = normalizedValue;
    if (this.metadata.targetNormalization) {
      const { min, max } = this.metadata.targetNormalization;
      const range = max - min;
      if (range > 0) {
        denormalizedValue = normalizedValue * range + min;
      }
    }
    
    return {
      taskType: 'regression',
      value: denormalizedValue,
    };
  }
} 