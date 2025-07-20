/**
 * @fileoverview Trainer implementation for complete ML training pipeline
 * 
 * The Trainer class handles the complete training pipeline from data loading
 * to model persistence, including feature encoding, XGBoost training,
 * cross-validation, and multi-class classification support.
 */

import { XGBoost } from '@fractal-solutions/xgboost-js';
import { HashEncoder } from './HashEncoder';
import { FeatureAnalyzer } from './FeatureAnalyzer';
import { DataLoader } from '../utils/DataLoader';
import { TreeVisualizer } from '../utils/TreeVisualizer';
import {
  RawDataRecord,
  Dataset,
  DataSplit,
  TrainingConfig,
  XgbParams,
  ModelMetadata,
  ModelMetrics,
  Logger,
  PredictionResult,
  TrainingProgressCallback,
  TaskType,
  TreeVisualizationOptions,
  VisualizationFormat,
  TreeVisualizationResult,
} from '../types';

/**
 * Trainer class for complete ML training pipeline
 * 
 * Features:
 * - Data loading from CSV, JSON, or raw arrays
 * - Automatic feature analysis and encoding
 * - XGBoost training with multi-class support
 * - Cross-validation and model evaluation
 * - One-vs-Rest fallback for multi-class problems
 * - Model persistence with metadata
 * - Support for both classification and regression tasks
 * 
 * @example
 * ```typescript
 * // Classification
 * const trainer = new Trainer({
 *   categoricalFeatures: ['brand', 'color'],
 *   numericFeatures: ['price', 'rating'],
 *   target: 'sold',
 *   taskType: 'classification',
 *   xgbParams: {
 *     objective: 'binary:logistic',
 *     max_depth: 6,
 *     eta: 0.3,
 *   },
 * });
 * 
 * // Regression
 * const regressionTrainer = new Trainer({
 *   categoricalFeatures: ['brand', 'color'],
 *   numericFeatures: ['area', 'age'],
 *   target: 'price',
 *   taskType: 'regression',
 *   xgbParams: {
 *     objective: 'reg:squarederror',
 *     max_depth: 6,
 *     eta: 0.3,
 *   },
 * });
 * ```
 */
export class Trainer {
  private categoricalFeatures: string[];
  private numericFeatures: string[];
  private target: string;
  private taskType: TaskType;
  private xgbParams: XgbParams;
  private logger: Logger | undefined;
  
  private dataset: Dataset | undefined;
  private featureAnalyzer: FeatureAnalyzer;
  private dataLoader: DataLoader;
  private encoders: Map<string, HashEncoder> = new Map();
  private model: XGBoost | undefined;
  private classes: string[] = [];
  private isMultiClass: boolean = false;
  private isOneVsRest: boolean = false;
  private oneVsRestModels: Map<string, XGBoost> = new Map();
  private trainedAt: string | undefined;
  private trainingMetrics: ModelMetrics | undefined;
  private targetStats: { min: number; max: number; mean: number; std: number; } | undefined;
  
  // Add normalization parameters for regression
  private targetNormalization: { min: number; max: number; } | undefined;

  /**
   * Creates a new Trainer instance
   * 
   * @param config - Training configuration
   * @param logger - Optional logger for debugging
   */
  constructor(config: {
    categoricalFeatures: string[];
    numericFeatures: string[];
    target: string;
    taskType?: TaskType;
    xgbParams: XgbParams;
  }, logger?: Logger) {
    this.categoricalFeatures = config.categoricalFeatures;
    this.numericFeatures = config.numericFeatures;
    this.target = config.target;
    this.taskType = config.taskType || 'classification'; // Default to classification for backward compatibility
    this.xgbParams = config.xgbParams;
    this.logger = logger;
    
    this.featureAnalyzer = new FeatureAnalyzer(logger);
    this.dataLoader = new DataLoader(logger);
    
    this.logger?.info('Trainer initialized', {
      categoricalFeatures: this.categoricalFeatures.length,
      numericFeatures: this.numericFeatures.length,
      target: this.target,
      taskType: this.taskType,
    });
  }

  /**
   * Loads data from a CSV file
   * 
   * @param filePath - Path to the CSV file
   * @param config - Optional CSV loading configuration
   */
  public async loadCSV(filePath: string, config?: any): Promise<void> {
    this.logger?.info('Loading data from CSV', { filePath });
    
    const data = await this.dataLoader.loadCSV(filePath, config);
    await this.loadRaw(data);
  }

  /**
   * Loads data from a JSON file
   * 
   * @param filePath - Path to the JSON file
   */
  public async loadJSON(filePath: string): Promise<void> {
    this.logger?.info('Loading data from JSON', { filePath });
    
    const data = await this.dataLoader.loadJSON(filePath);
    await this.loadRaw(data);
  }

  /**
   * Loads raw data array
   * 
   * @param data - Array of raw data records
   */
  public async loadRaw(data: RawDataRecord[]): Promise<void> {
    this.logger?.info('Loading raw data', { records: data.length });
    
    // Validate data
    const allFeatures = [...this.categoricalFeatures, ...this.numericFeatures, this.target];
    if (!this.dataLoader.validateData(data, allFeatures)) {
      throw new Error('Data validation failed');
    }
    
    // Analyze features
    const analysis = this.featureAnalyzer.analyzeFeatures(
      data,
      this.categoricalFeatures,
      this.numericFeatures,
      this.target
    );
    
    // Create hash encoders for categorical features
    for (const [featureName, spec] of Object.entries(analysis.categoricalSpecs)) {
      const encoder = new HashEncoder(spec.bucketCount, 42, featureName);
      this.encoders.set(featureName, encoder);
    }
    
    if (this.taskType === 'classification') {
      // Determine classes and multi-class setup for classification
      this.classes = analysis.datasetStats.targetClasses.sort();
      this.isMultiClass = this.classes.length > 2;
      
      this.logger?.info('Data loaded successfully (classification)', {
        classes: this.classes.length,
        isMultiClass: this.isMultiClass,
        encoders: this.encoders.size,
      });
    } else {
      // For regression, calculate target statistics
      this.targetStats = this.calculateTargetStats(data);
      
      // Calculate normalization parameters for regression
      this.targetNormalization = {
        min: this.targetStats.min,
        max: this.targetStats.max,
      };
      
      this.logger?.info('Data loaded successfully (regression)', {
        targetStats: this.targetStats,
        targetNormalization: this.targetNormalization,
        encoders: this.encoders.size,
      });
    }
    
    // Store dataset
    this.dataset = {
      data,
      categoricalFeatures: this.categoricalFeatures,
      numericFeatures: this.numericFeatures,
      target: this.target,
    };
  }

  /**
   * Trains the model with specified configuration
   * 
   * @param config - Training configuration
   * @param progressCallback - Optional progress callback
   */
  public async train(
    config: TrainingConfig = {},
    progressCallback?: TrainingProgressCallback
  ): Promise<void> {
    if (!this.dataset) {
      throw new Error('No data loaded. Call loadCSV, loadJSON, or loadRaw first.');
    }
    
    this.logger?.info('Starting training', config);
    const startTime = performance.now();
    
    // Configure XGBoost parameters based on problem type
    const xgbParams = this.configureXgbParams();
    
    // Split data if needed
    const dataSplit = this.splitData(this.dataset, config);
    
    // Transform features
    const { X_train, y_train, X_test, y_test } = this.transformFeatures(dataSplit);
    
    if (this.isMultiClass && this.shouldUseOneVsRest()) {
      // Train One-vs-Rest models
      await this.trainOneVsRest(X_train, y_train, xgbParams, progressCallback);
    } else {
      // Train single model
      await this.trainSingleModel(X_train, y_train, xgbParams, progressCallback);
    }
    
    // Evaluate model
    if (X_test && y_test) {
      this.trainingMetrics = await this.evaluateModel(X_test, y_test);
    }
    
    // Perform cross-validation if requested
    if (config.useCrossValidation) {
      await this.performCrossValidation(config);
    }
    
    const endTime = performance.now();
    this.trainedAt = new Date().toISOString();
    
    this.logger?.info('Training completed', {
      duration: `${(endTime - startTime).toFixed(2)}ms`,
      isOneVsRest: this.isOneVsRest,
      metrics: this.trainingMetrics,
    });
  }

  /**
   * Saves the trained model to a directory
   * 
   * @param directory - Directory to save the model
   */
  public async save(directory: string): Promise<void> {
    if (!this.model && this.oneVsRestModels.size === 0) {
      throw new Error('No trained model to save. Call train() first.');
    }
    
    this.logger?.info('Saving model', { directory });
    
    // Create directory if it doesn't exist
    const fs = require('fs');
    if (!fs.existsSync(directory)) {
      fs.mkdirSync(directory, { recursive: true });
    }
    
    // Save model(s)
    if (this.isOneVsRest) {
      // Save One-vs-Rest models
      const modelsDir = `${directory}/models`;
      if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir, { recursive: true });
      }
      
      for (const [className, model] of this.oneVsRestModels.entries()) {
        const modelJSON = model.toJSON();
        fs.writeFileSync(`${modelsDir}/${className}.json`, JSON.stringify(modelJSON, null, 2));
      }
    } else {
      // Save single model
      if (this.model) {
        const modelJSON = this.model.toJSON();
        fs.writeFileSync(`${directory}/model.json`, JSON.stringify(modelJSON, null, 2));
      }
    }
    
    // Save encoders
    const encodersConfig = Array.from(this.encoders.entries()).map(([, encoder]) => ({
      ...encoder.getConfig(),
    }));
    fs.writeFileSync(`${directory}/encoders.json`, JSON.stringify(encodersConfig, null, 2));
    
    // Save metadata
    const metadata = this.getModelMetadata();
    fs.writeFileSync(`${directory}/metadata.json`, JSON.stringify(metadata, null, 2));
    
    this.logger?.info('Model saved successfully', { directory });
  }

  /**
   * Returns the model's training metrics
   */
  public evaluate(): ModelMetrics {
    if (!this.trainingMetrics) {
      throw new Error('No training metrics available. Train the model first.');
    }
    return this.trainingMetrics;
  }

  /**
   * Returns metadata about the trained model
   */
  public getModelMetadata(): ModelMetadata {
    if (!this.trainedAt) {
      throw new Error('Model not trained yet');
    }

    const metadata: ModelMetadata = {
      version: '1.0.0',
      createdAt: this.trainedAt,
      categoricalFeatures: this.categoricalFeatures,
      numericFeatures: this.numericFeatures,
      target: this.target,
      taskType: this.taskType,
      xgbParams: this.xgbParams,
      encoders: Array.from(this.encoders.values()).map(encoder => ({
        ...encoder.getConfig(),
        uniqueCount: 0, // Will be populated during training
      })),
      metrics: this.trainingMetrics!,
      isOneVsRest: this.isOneVsRest,
    };

    if (this.taskType === 'classification') {
      metadata.classes = this.classes;
      metadata.numClasses = this.classes.length;
    } else {
      // Include regression-specific metadata
      if (this.targetStats) {
        metadata.targetStats = this.targetStats;
      }
      if (this.targetNormalization) {
        metadata.targetNormalization = this.targetNormalization;
      }
    }

    return metadata;
  }

  /**
   * Visualizes a decision tree from the trained model
   * 
   * @param format - Visualization format ('text', 'json', 'html', 'svg')
   * @param options - Visualization options
   * @returns Tree visualization result
   * 
   * @example
   * ```typescript
   * // Text visualization
   * const textViz = trainer.visualizeTree('text', { treeIndex: 0, includeFeatureNames: true });
   * console.log(textViz.content);
   * 
   * // HTML visualization
   * const htmlViz = trainer.visualizeTree('html', { maxDepth: 3, precision: 2 });
   * ```
   */
  public visualizeTree(
    format: VisualizationFormat,
    options: TreeVisualizationOptions = {}
  ): TreeVisualizationResult {
    if (!this.model && this.oneVsRestModels.size === 0) {
      throw new Error('No trained model available. Call train() first.');
    }

    // Create feature names array for the visualizer
    const featureNames: string[] = [];
    
    // Add categorical features (expanded for hash encoding)
    for (const featureName of this.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        const dimension = encoder.getDimension();
        for (let i = 0; i < dimension; i++) {
          featureNames.push(`${featureName}_${i}`);
        }
      }
    }
    
    // Add numeric features
    featureNames.push(...this.numericFeatures);

    const visualizer = new TreeVisualizer(featureNames, this.logger);

    if (this.isOneVsRest) {
      // For One-vs-Rest, visualize a specific class model
      if (this.classes.length === 0) {
        throw new Error('No classes available for One-vs-Rest model');
      }
      
      const className = options.className || this.classes[0]!;
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
   * // Save HTML visualization
   * await trainer.saveTreeVisualization('./tree.html', 'html', { includeFeatureNames: true });
   * 
   * // Save SVG visualization
   * await trainer.saveTreeVisualization('./tree.svg', 'svg', { maxDepth: 5 });
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
   * Gets tree statistics from the trained model
   * 
   * @returns Tree statistics for all trees in the model
   * 
   * @example
   * ```typescript
   * const stats = trainer.getTreeStatistics();
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
      throw new Error('No trained model available. Call train() first.');
    }

    const featureNames: string[] = [];
    
    // Add categorical features (expanded for hash encoding)
    for (const featureName of this.categoricalFeatures) {
      const encoder = this.encoders.get(featureName);
      if (encoder) {
        const dimension = encoder.getDimension();
        for (let i = 0; i < dimension; i++) {
          featureNames.push(`${featureName}_${i}`);
        }
      }
    }
    
    // Add numeric features
    featureNames.push(...this.numericFeatures);

    const visualizer = new TreeVisualizer(featureNames, this.logger);

    if (this.isOneVsRest) {
      // For One-vs-Rest, return statistics for first class model
      if (this.classes.length === 0) {
        throw new Error('No classes available for One-vs-Rest model');
      }
      
      const className = this.classes[0]!;
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
   * Configures XGBoost parameters based on task type and problem characteristics
   */
  private configureXgbParams(): XgbParams {
    const params = { ...this.xgbParams };

    if (this.taskType === 'classification') {
      // Configure for classification
      if (this.isMultiClass && this.classes.length > 2) {
        params.objective = params.objective || 'multi:softprob';
        params.num_class = this.classes.length;
      } else {
        params.objective = params.objective || 'binary:logistic';
      }
    } else {
      // Configure for regression
      params.objective = params.objective || 'reg:squarederror';
      // Remove num_class for regression
      delete params.num_class;
    }

    return params;
  }

  /**
   * Splits data into training and test sets
   * 
   * @private
   */
  private splitData(dataset: Dataset, config: TrainingConfig): DataSplit {
    const { data } = dataset;
    const testRatio = config.testRatio || 0.2;
    const shuffle = config.shuffle !== false;
    
    let shuffledData = [...data];
    if (shuffle) {
      // Simple shuffle
      for (let i = shuffledData.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffledData[i], shuffledData[j]] = [shuffledData[j]!, shuffledData[i]!];
      }
    }
    
    const splitIndex = Math.floor(data.length * (1 - testRatio));
    const trainData = shuffledData.slice(0, splitIndex);
    const testData = shuffledData.slice(splitIndex);
    
    return {
      train: { ...dataset, data: trainData },
      test: { ...dataset, data: testData },
    };
  }

  /**
   * Transforms features using encoders
   * 
   * @private
   */
  private transformFeatures(dataSplit: DataSplit): {
    X_train: number[][];
    y_train: number[];
    X_test?: number[][];
    y_test?: number[];
  } {
    const X_train = this.transformRecords(dataSplit.train.data);
    const y_train = this.transformTargets(dataSplit.train.data);
    
    let X_test: number[][] | undefined;
    let y_test: number[] | undefined;
    
    if (dataSplit.test.data.length > 0) {
      X_test = this.transformRecords(dataSplit.test.data);
      y_test = this.transformTargets(dataSplit.test.data);
    }
    
    return { 
      X_train, 
      y_train, 
      ...(X_test && { X_test }), 
      ...(y_test && { y_test }) 
    };
  }

  /**
   * Transforms data records to feature vectors
   * 
   * @private
   */
  private transformRecords(records: RawDataRecord[]): number[][] {
    return records.map(record => {
      const features: number[] = [];
      
      // Add categorical features (encoded)
      for (const featureName of this.categoricalFeatures) {
        const encoder = this.encoders.get(featureName);
        if (encoder) {
          const encoded = encoder.encode(record[featureName]);
          features.push(...Array.from(encoded));
        }
      }
      
      // Add numerical features
      for (const featureName of this.numericFeatures) {
        const value = record[featureName];
        features.push(typeof value === 'number' ? value : 0);
      }
      
      return features;
    });
  }

  /**
   * Transforms target values to numeric indices for classification or keeps as numeric for regression
   */
  private transformTargets(records: RawDataRecord[]): number[] {
    if (this.taskType === 'classification') {
      return records.map(record => {
        const targetValue = String(record[this.target]);
        const classIndex = this.classes.indexOf(targetValue);
        if (classIndex === -1) {
          throw new Error(`Unknown class: ${targetValue}`);
        }
        return classIndex;
      });
    } else {
      // For regression, convert to numeric values and normalize to 0-1 range
      if (!this.targetNormalization) {
        throw new Error('Target normalization parameters not set for regression');
      }
      
      const range = this.targetNormalization.max - this.targetNormalization.min;
      if (range === 0) {
        // All target values are the same, return array of 0.5
        return records.map(() => 0.5);
      }
      
      return records.map(record => {
        const targetValue = record[this.target];
        let numericValue: number;
        
        if (typeof targetValue === 'number') {
          numericValue = targetValue;
        } else if (typeof targetValue === 'string') {
          numericValue = parseFloat(targetValue);
          if (isNaN(numericValue)) {
            throw new Error(`Invalid numeric target value: ${targetValue}`);
          }
        } else {
          throw new Error(`Target value must be numeric for regression: ${targetValue}`);
        }
        
        // Normalize to 0-1 range
        const normalizedValue = (numericValue - this.targetNormalization!.min) / range;
        return normalizedValue;
      });
    }
  }

  /**
   * Determines if One-vs-Rest should be used for multi-class classification
   */
  private shouldUseOneVsRest(): boolean {
    if (this.taskType === 'regression') {
      return false; // One-vs-Rest only applies to classification
    }
    
    // Use One-vs-Rest for multi-class if XGBoost doesn't support multi:softprob
    // This is a fallback mechanism
    return this.isMultiClass && !this.xgbParams.objective?.startsWith('multi:');
  }

  /**
   * Trains a single XGBoost model
   * 
   * @private
   */
  private async trainSingleModel(
    X_train: number[][],
    y_train: number[],
    xgbParams: XgbParams,
    _progressCallback?: TrainingProgressCallback
  ): Promise<void> {
    // Set task-specific defaults
    const isRegression = this.taskType === 'regression';
    
    const params = {
      learningRate: xgbParams.eta || (isRegression ? 0.05 : 0.1),
      maxDepth: xgbParams.max_depth || (isRegression ? 5 : 6),
      numRounds: xgbParams.num_boost_round || (isRegression ? 100 : 50),
      objective: xgbParams.objective || (isRegression ? 'reg:squarederror' : 'binary:logistic'),
      evalMetric: xgbParams.eval_metric || (isRegression ? 'rmse' : 'auc'),
      seed: xgbParams.seed || 42,
      minChildWeight: xgbParams.min_child_weight || 1,
      booster: xgbParams.booster || 'gbtree',
      subsample: xgbParams.subsample || (isRegression ? 0.8 : 1.0),
      colsampleBytree: xgbParams.colsample_bytree || (isRegression ? 0.8 : 1.0),
      verbosity: xgbParams.verbosity || 1,
      gamma: xgbParams.gamma || 0,
      alpha: xgbParams.alpha || 0,
      lambda: xgbParams.lambda || 1,
      nthread: xgbParams.nthread || -1,
    };
    
    this.model = new XGBoost(params);
    
    // Use the fit method instead of train
    this.model.fit(X_train, y_train);
    this.isOneVsRest = false;
  }

  /**
   * Trains One-vs-Rest models for multi-class classification
   * 
   * @private
   */
  private async trainOneVsRest(
    X_train: number[][],
    y_train: number[],
    xgbParams: XgbParams,
    _progressCallback?: TrainingProgressCallback
  ): Promise<void> {
    this.isOneVsRest = true;
    
    // Train binary classifier for each class
    for (let classIdx = 0; classIdx < this.classes.length; classIdx++) {
      const className = this.classes[classIdx];
      
      // Create binary labels (1 for current class, 0 for others)
      const binaryLabels = y_train.map(label => (label === classIdx ? 1 : 0));
      
      // Train binary classifier with classification defaults
      const params = {
        learningRate: xgbParams.eta || 0.1,
        maxDepth: xgbParams.max_depth || 6,
        numRounds: xgbParams.num_boost_round || 50,
        objective: 'binary:logistic', // Always binary for One-vs-Rest
        evalMetric: xgbParams.eval_metric || 'auc',
        seed: xgbParams.seed || 42,
        minChildWeight: xgbParams.min_child_weight || 1,
        booster: xgbParams.booster || 'gbtree',
        subsample: xgbParams.subsample || 1.0,
        colsampleBytree: xgbParams.colsample_bytree || 1.0,
        verbosity: xgbParams.verbosity || 1,
        gamma: xgbParams.gamma || 0,
        alpha: xgbParams.alpha || 0,
        lambda: xgbParams.lambda || 1,
        nthread: xgbParams.nthread || -1,
      };
      
      const model = new XGBoost(params);
      model.fit(X_train, binaryLabels);
      
      this.oneVsRestModels.set(className!, model);
    }
  }

  /**
   * Evaluates the model performance on test data
   */
  private async evaluateModel(X_test: number[][], y_test: number[]): Promise<ModelMetrics> {
    const predictions = await this.predictBatch(X_test);
    
    if (this.taskType === 'classification') {
      // Use existing classification evaluation
      const predicted = predictions.map(p => p.classIndex!);
      
      return this.evaluateClassification(predicted, y_test);
    } else {
      // Use regression evaluation
      const predicted = predictions.map(p => p.value!);
      
      return this.evaluateRegression(predicted, y_test);
    }
  }

  /**
   * Evaluates classification metrics
   */
  private evaluateClassification(predicted: number[], actual: number[]): ModelMetrics {
    const accuracy = predicted.filter((pred, idx) => pred === actual[idx]).length / predicted.length;
    const confusionMatrix = this.calculateConfusionMatrix(predicted, actual);
    
    // Calculate precision, recall, and F1-score for each class
    const precision: number[] = [];
    const recall: number[] = [];
    const f1Score: number[] = [];
    
    for (let classIdx = 0; classIdx < this.classes.length; classIdx++) {
      const tp = predicted.filter((pred, idx) => pred === classIdx && actual[idx] === classIdx).length;
      const fp = predicted.filter((pred, idx) => pred === classIdx && actual[idx] !== classIdx).length;
      const fn = predicted.filter((pred, idx) => pred !== classIdx && actual[idx] === classIdx).length;
      
      const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
      const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
      
      precision.push(prec);
      recall.push(rec);
      f1Score.push(f1);
    }
    
    return {
      taskType: 'classification',
      accuracy,
      confusionMatrix,
      precision,
      recall,
      f1Score,
    };
  }

  /**
   * Evaluates regression metrics
   */
  private evaluateRegression(predicted: number[], actual: number[]): ModelMetrics {
    const n = predicted.length;
    
    // Mean Squared Error
    const mse = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.pow(pred - actualVal, 2) : 0);
    }, 0) / n;
    
    // Root Mean Squared Error
    const rmse = Math.sqrt(mse);
    
    // Mean Absolute Error
    const mae = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.abs(pred - actualVal) : 0);
    }, 0) / n;
    
    // R-squared (coefficient of determination)
    const validActual = actual.filter(val => val !== undefined);
    const actualMean = validActual.reduce((sum, val) => sum + val, 0) / validActual.length;
    const ssRes = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.pow(actualVal - pred, 2) : 0);
    }, 0);
    const ssTot = validActual.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
    const r2 = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    
    // Mean Absolute Percentage Error
    const mape = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      if (actualVal !== undefined && actualVal !== 0) {
        return sum + Math.abs((actualVal - pred) / actualVal);
      }
      return sum;
    }, 0) / n * 100;
    
    // Median Absolute Error
    const absoluteErrors = predicted.map((pred, idx) => {
      const actualVal = actual[idx];
      return actualVal !== undefined ? Math.abs(pred - actualVal) : 0;
    }).sort((a, b) => a - b);
    const medianAe = absoluteErrors.length > 0 ? absoluteErrors[Math.floor(absoluteErrors.length / 2)]! : 0;
    
    return {
      taskType: 'regression',
      mse,
      rmse,
      mae,
      r2,
      mape,
      medianAe,
    };
  }

  /**
   * Calculates confusion matrix
   * 
   * @private
   */
  private calculateConfusionMatrix(predicted: number[], actual: number[]): number[][] {
    const matrix = Array(this.classes.length).fill(0).map(() => Array(this.classes.length).fill(0));
    
    for (let i = 0; i < predicted.length; i++) {
      matrix[actual[i]!]![predicted[i]!]++;
    }
    
    return matrix;
  }

  /**
   * Calculates target statistics for regression
   */
  private calculateTargetStats(data: RawDataRecord[]): { min: number; max: number; mean: number; std: number; } {
    const targetValues = data.map(record => {
      const value = record[this.target];
      if (typeof value === 'number') {
        return value;
      }
      if (typeof value === 'string') {
        const numericValue = parseFloat(value);
        if (isNaN(numericValue)) {
          throw new Error(`Invalid numeric target value: ${value}`);
        }
        return numericValue;
      }
      throw new Error(`Target value must be numeric for regression: ${value}`);
    });
    
    const min = Math.min(...targetValues);
    const max = Math.max(...targetValues);
    const mean = targetValues.reduce((sum, val) => sum + val, 0) / targetValues.length;
    const variance = targetValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / targetValues.length;
    const std = Math.sqrt(variance);
    
    return { min, max, mean, std };
  }

  /**
   * Predicts classes or values for a batch of feature vectors
   */
  private async predictBatch(X: number[][]): Promise<PredictionResult[]> {
    if (this.taskType === 'classification') {
      return this.predictClassificationBatch(X);
    } else {
      return this.predictRegressionBatch(X);
    }
  }

  /**
   * Predicts classes for a batch of feature vectors (classification)
   */
  private async predictClassificationBatch(X: number[][]): Promise<PredictionResult[]> {
    if (this.isOneVsRest) {
      // One-vs-Rest prediction
      const results: PredictionResult[] = [];
      
      for (const features of X) {
        const classScores = new Map<string, number>();
        
        for (const [className, model] of this.oneVsRestModels.entries()) {
          try {
            // Use predictSingle for single prediction
            const prediction = model.predictSingle(features);
            classScores.set(className, prediction);
          } catch (error) {
            this.logger?.warn(`Prediction failed for class ${className}:`, error);
            classScores.set(className, 0);
          }
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
        
        const classIndex = this.classes.indexOf(bestClass);
        const probabilities = this.classes.map(cls => classScores.get(cls) || 0);
        
        results.push({
          taskType: 'classification',
          classIndex,
          classLabel: bestClass,
          probability: bestScore,
          probabilities,
        });
      }
      
      return results;
    } else {
      // Single model prediction
      if (!this.model) {
        throw new Error('Model not trained');
      }
      
      const predictions = this.model.predictBatch(X);
      
      return predictions.map((pred: number) => {
        if (this.isMultiClass) {
          // Multi-class: for XGBoost, we might need to handle this differently
          // For now, treat single prediction as binary
          const probability = pred;
          const classIndex = probability > 0.5 ? 1 : 0;
          
          return {
            taskType: 'classification',
            classIndex,
            classLabel: this.classes[classIndex] || 'unknown',
            probability,
            probabilities: [1 - probability, probability],
          };
        } else {
          // Binary classification: pred is single probability
          const probability = pred;
          const classIndex = probability > 0.5 ? 1 : 0;
          
          return {
            taskType: 'classification',
            classIndex,
            classLabel: this.classes[classIndex] || 'unknown',
            probability,
            probabilities: [1 - probability, probability],
          };
        }
      });
    }
  }

  /**
   * Predicts values for a batch of feature vectors (regression)
   */
  private async predictRegressionBatch(X: number[][]): Promise<PredictionResult[]> {
    if (!this.model) {
      throw new Error('Model not trained');
    }
    
    const predictions = this.model.predictBatch(X);
    
    return predictions.map((pred: number) => ({
      taskType: 'regression',
      value: pred,
    }));
  }

  /**
   * Performs cross-validation
   * 
   * @private
   */
  private async performCrossValidation(config: TrainingConfig): Promise<void> {
    if (!this.dataset || !config.useCrossValidation) {
      return;
    }
    
    const nFolds = config.nFolds || 5;
    const foldSize = Math.floor(this.dataset.data.length / nFolds);
    const scores: number[] = [];
    
    this.logger?.info(`Starting ${nFolds}-fold cross-validation`);
    
    for (let fold = 0; fold < nFolds; fold++) {
      // Create fold split
      const testStart = fold * foldSize;
      const testEnd = fold === nFolds - 1 ? this.dataset.data.length : testStart + foldSize;
      
      const testData = this.dataset.data.slice(testStart, testEnd);
      const trainData = [
        ...this.dataset.data.slice(0, testStart),
        ...this.dataset.data.slice(testEnd),
      ];
      
      // Transform features for this fold
      const X_train = this.transformRecords(trainData);
      const y_train = this.transformTargets(trainData);
      const X_test = this.transformRecords(testData);
      const y_test = this.transformTargets(testData);
      
      // Train temporary model
      const xgbParams = this.configureXgbParams();
      
      // Set task-specific defaults for cross-validation
      const isRegression = this.taskType === 'regression';
      
      const params = {
        learningRate: xgbParams.eta || (isRegression ? 0.05 : 0.1),
        maxDepth: xgbParams.max_depth || (isRegression ? 5 : 6),
        numRounds: xgbParams.num_boost_round || (isRegression ? 100 : 50),
        objective: xgbParams.objective || (isRegression ? 'reg:squarederror' : 'binary:logistic'),
        evalMetric: xgbParams.eval_metric || (isRegression ? 'rmse' : 'auc'),
        seed: xgbParams.seed || 42,
        minChildWeight: xgbParams.min_child_weight || 1,
        booster: xgbParams.booster || 'gbtree',
        subsample: xgbParams.subsample || (isRegression ? 0.8 : 1.0),
        colsampleBytree: xgbParams.colsample_bytree || (isRegression ? 0.8 : 1.0),
        verbosity: xgbParams.verbosity || 1,
        gamma: xgbParams.gamma || 0,
        alpha: xgbParams.alpha || 0,
        lambda: xgbParams.lambda || 1,
        nthread: xgbParams.nthread || -1,
      };
      
      const tempModel = new XGBoost(params);
      tempModel.fit(X_train, y_train);
      
      // Evaluate on test set
      const predictions = X_test.map(x => tempModel.predictSingle(x));
      
      let score: number;
      if (this.taskType === 'classification') {
        // Use accuracy for classification
        const correct = predictions.filter((pred, idx) => {
          const predicted = pred > 0.5 ? 1 : 0;
          return predicted === y_test[idx];
        }).length;
        score = correct / predictions.length;
      } else {
        // Use RMSE for regression
        const mse = predictions.reduce((sum, pred, idx) => {
          const actualVal = y_test[idx];
          return sum + (actualVal !== undefined ? Math.pow(pred - actualVal, 2) : 0);
        }, 0) / predictions.length;
        score = Math.sqrt(mse);
      }
      
      scores.push(score);
      this.logger?.info(`Fold ${fold + 1}/${nFolds} - Score: ${score.toFixed(4)}`);
    }
    
    // Update metrics with CV scores
    if (this.trainingMetrics) {
      this.trainingMetrics.crossValidationScores = scores;
    }
    
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const scoreType = this.taskType === 'classification' ? 'accuracy' : 'RMSE';
    this.logger?.info(`Cross-validation completed - Average ${scoreType}: ${avgScore.toFixed(4)}`);
  }
} 