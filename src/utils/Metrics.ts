/**
 * @fileoverview Metrics utility for model evaluation
 * 
 * Provides comprehensive evaluation metrics for classification and regression problems
 * including accuracy, precision, recall, F1-score, ROC AUC, confusion matrix for classification,
 * and MSE, RMSE, MAE, R² for regression.
 */

import { ModelMetrics, RegressionMetrics } from '../types';

/**
 * Metrics utility class for model evaluation
 * 
 * Features:
 * - Classification accuracy calculation
 * - Precision, recall, and F1-score per class
 * - ROC AUC for binary and multi-class problems
 * - Confusion matrix generation
 * - Cross-validation score aggregation
 * - Statistical significance testing
 * - Regression metrics (MSE, RMSE, MAE, R²)
 * 
 * @example
 * ```typescript
 * // Classification metrics
 * const predicted = [0, 1, 1, 0, 1];
 * const actual = [0, 1, 0, 0, 1];
 * 
 * const accuracy = Metrics.accuracy(predicted, actual);
 * const precision = Metrics.precision(predicted, actual);
 * const recall = Metrics.recall(predicted, actual);
 * const f1 = Metrics.f1Score(predicted, actual);
 * 
 * // Regression metrics
 * const predictedValues = [2.5, 3.2, 1.8, 4.1];
 * const actualValues = [2.0, 3.5, 2.1, 4.0];
 * 
 * const mse = Metrics.meanSquaredError(predictedValues, actualValues);
 * const rmse = Metrics.rootMeanSquaredError(predictedValues, actualValues);
 * const mae = Metrics.meanAbsoluteError(predictedValues, actualValues);
 * const r2 = Metrics.r2Score(predictedValues, actualValues);
 * ```
 */
export class Metrics {
  /**
   * Calculates classification accuracy
   * 
   * @param predicted - Array of predicted class indices
   * @param actual - Array of actual class indices
   * @returns Accuracy score (0-1)
   * 
   * @example
   * ```typescript
   * const accuracy = Metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]);
   * console.log('Accuracy:', accuracy); // 0.75
   * ```
   */
  public static accuracy(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    const correct = predicted.filter((pred, idx) => pred === actual[idx]).length;
    return correct / predicted.length;
  }

  /**
   * Calculates precision for each class
   * 
   * @param predicted - Array of predicted class indices
   * @param actual - Array of actual class indices
   * @param numClasses - Number of classes (auto-detected if not provided)
   * @returns Array of precision scores per class
   * 
   * @example
   * ```typescript
   * const precision = Metrics.precision([0, 1, 1, 0], [0, 1, 0, 0]);
   * console.log('Precision per class:', precision); // [0.67, 0.5]
   * ```
   */
  public static precision(predicted: number[], actual: number[], numClasses?: number): number[] {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    const classes = numClasses || Math.max(...predicted, ...actual) + 1;
    const precision: number[] = [];
    
    for (let classIdx = 0; classIdx < classes; classIdx++) {
      const tp = predicted.filter((pred, idx) => pred === classIdx && actual[idx] === classIdx).length;
      const fp = predicted.filter((pred, idx) => pred === classIdx && actual[idx] !== classIdx).length;
      
      precision.push(tp + fp > 0 ? tp / (tp + fp) : 0);
    }
    
    return precision;
  }

  /**
   * Calculates recall for each class
   * 
   * @param predicted - Array of predicted class indices
   * @param actual - Array of actual class indices
   * @param numClasses - Number of classes (auto-detected if not provided)
   * @returns Array of recall scores per class
   * 
   * @example
   * ```typescript
   * const recall = Metrics.recall([0, 1, 1, 0], [0, 1, 0, 0]);
   * console.log('Recall per class:', recall); // [0.67, 1.0]
   * ```
   */
  public static recall(predicted: number[], actual: number[], numClasses?: number): number[] {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    const classes = numClasses || Math.max(...predicted, ...actual) + 1;
    const recall: number[] = [];
    
    for (let classIdx = 0; classIdx < classes; classIdx++) {
      const tp = predicted.filter((pred, idx) => pred === classIdx && actual[idx] === classIdx).length;
      const fn = predicted.filter((pred, idx) => pred !== classIdx && actual[idx] === classIdx).length;
      
      recall.push(tp + fn > 0 ? tp / (tp + fn) : 0);
    }
    
    return recall;
  }

  /**
   * Calculates F1-score for each class
   * 
   * @param predicted - Array of predicted class indices
   * @param actual - Array of actual class indices
   * @param numClasses - Number of classes (auto-detected if not provided)
   * @returns Array of F1 scores per class
   * 
   * @example
   * ```typescript
   * const f1 = Metrics.f1Score([0, 1, 1, 0], [0, 1, 0, 0]);
   * console.log('F1 per class:', f1); // [0.67, 0.67]
   * ```
   */
  public static f1Score(predicted: number[], actual: number[], numClasses?: number): number[] {
    const precision = this.precision(predicted, actual, numClasses);
    const recall = this.recall(predicted, actual, numClasses);
    
    return precision.map((prec, idx) => {
      const rec = recall[idx];
      return prec + (rec || 0) > 0 ? (2 * prec * (rec || 0)) / (prec + (rec || 0)) : 0;
    });
  }

  /**
   * Calculates confusion matrix
   * 
   * @param actual - Actual class labels
   * @param predicted - Predicted class labels
   * @param classes - Number of classes
   * @returns Confusion matrix
   */
  public static confusionMatrix(actual: number[], predicted: number[], classes: number): number[][] {
    const matrix = Array(classes).fill(0).map(() => Array(classes).fill(0));
    
    for (let i = 0; i < actual.length; i++) {
      const actualClass = actual[i];
      const predictedClass = predicted[i];
      
      // Add type guards to ensure indices are valid
      if (actualClass !== undefined && predictedClass !== undefined && 
          actualClass >= 0 && actualClass < classes && 
          predictedClass >= 0 && predictedClass < classes) {
        matrix[actualClass]![predictedClass]++;
      }
    }
    
    return matrix;
  }

  /**
   * Calculates ROC AUC for binary classification
   * 
   * @param probabilities - Array of predicted probabilities for positive class
   * @param actual - Array of actual binary labels (0 or 1)
   * @returns ROC AUC score
   * 
   * @example
   * ```typescript
   * const rocAuc = Metrics.rocAuc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]);
   * console.log('ROC AUC:', rocAuc); // 0.75
   * ```
   */
  public static rocAuc(probabilities: number[], actual: number[]): number {
    if (probabilities.length !== actual.length) {
      throw new Error('Probabilities and actual arrays must have the same length');
    }
    
    // Sort by probabilities in descending order
    const pairs = probabilities.map((prob, idx) => ({ prob, actual: actual[idx] }))
      .sort((a, b) => b.prob - a.prob);
    
    let tp = 0;
    let fp = 0;
    const tpRates: number[] = [];
    const fpRates: number[] = [];
    
    const totalPositives = actual.filter(label => label === 1).length;
    const totalNegatives = actual.filter(label => label === 0).length;
    
    if (totalPositives === 0 || totalNegatives === 0) {
      return 0.5; // Random performance when only one class present
    }
    
    // Calculate TPR and FPR at each threshold
    for (const pair of pairs) {
      if (pair.actual === 1) {
        tp++;
      } else {
        fp++;
      }
      
      const tpr = tp / totalPositives;
      const fpr = fp / totalNegatives;
      
      tpRates.push(tpr);
      fpRates.push(fpr);
    }
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < fpRates.length; i++) {
      const deltaX = fpRates[i]! - fpRates[i - 1]!;
      const avgY = (tpRates[i]! + tpRates[i - 1]!) / 2;
      auc += deltaX * avgY;
    }
    
    return auc;
  }

  /**
   * Calculates macro-averaged ROC AUC for multi-class classification
   * 
   * @param probabilities - 2D array of predicted probabilities [samples][classes]
   * @param actual - Array of actual class indices
   * @returns Macro-averaged ROC AUC score
   * 
   * @example
   * ```typescript
   * const probabilities = [
   *   [0.7, 0.2, 0.1],
   *   [0.1, 0.8, 0.1],
   *   [0.2, 0.3, 0.5]
   * ];
   * const actual = [0, 1, 2];
   * const macroAuc = Metrics.macroRocAuc(probabilities, actual);
   * ```
   */
  public static macroRocAuc(probabilities: number[][], actual: number[]): number {
    if (probabilities.length !== actual.length) {
      throw new Error('Probabilities and actual arrays must have the same length');
    }
    
    if (probabilities.length === 0) {
      return 0.5;
    }
    
    const numClasses = probabilities[0]!.length;
    let sumAuc = 0;
    let validClasses = 0;
    
    for (let classIdx = 0; classIdx < numClasses; classIdx++) {
      // Create binary labels for one-vs-rest
      const binaryActual = actual.map(label => (label === classIdx ? 1 : 0));
      const classProbabilities = probabilities.map(probs => probs[classIdx] || 0);
      
      // Skip if class has no positive samples
      if (binaryActual.some(label => label === 1)) {
        const auc = this.rocAuc(classProbabilities, binaryActual);
        sumAuc += auc;
        validClasses++;
      }
    }
    
    return validClasses > 0 ? sumAuc / validClasses : 0.5;
  }

  /**
   * Calculates evaluation metrics for classification
   * 
   * @param predicted - Array of predicted class indices
   * @param actual - Array of actual class indices
   * @param probabilities - Array of probability arrays (optional)
   * @returns Model metrics
   */
  public static calculateMetrics(
    predicted: number[],
    actual: number[],
    probabilities?: number[][]
  ): ModelMetrics {
    const accuracy = this.accuracy(predicted, actual);
    const confusionMatrix = this.confusionMatrix(actual, predicted, Math.max(...actual, ...predicted) + 1);
    const precision = this.precision(predicted, actual);
    const recall = this.recall(predicted, actual);
    const f1Score = this.f1Score(predicted, actual);
    
    const metrics: ModelMetrics = {
      taskType: 'classification',
      accuracy,
      confusionMatrix,
      precision,
      recall,
      f1Score,
    };
    
    // Calculate ROC AUC if probabilities are provided
    if (probabilities) {
      if (probabilities.length > 0 && probabilities[0] && probabilities[0].length === 2) {
        // Binary classification
        const positiveProbabilities = probabilities.map(p => p[1] || 0);
        metrics.rocAuc = this.rocAuc(positiveProbabilities, actual);
      } else if (probabilities.length > 0 && probabilities[0] && probabilities[0].length > 2) {
        // Multi-class classification
        metrics.rocAuc = this.macroRocAuc(probabilities, actual);
      }
    }
    
    return metrics;
  }

  /**
   * Calculates mean and standard deviation of cross-validation scores
   * 
   * @param scores - Array of cross-validation scores
   * @returns Statistics object with mean, std, min, max
   * 
   * @example
   * ```typescript
   * const cvStats = Metrics.crossValidationStats([0.85, 0.87, 0.83, 0.86, 0.84]);
   * console.log('CV Mean:', cvStats.mean);
   * console.log('CV Std:', cvStats.std);
   * ```
   */
  public static crossValidationStats(scores: number[]): {
    mean: number;
    std: number;
    min: number;
    max: number;
    count: number;
  } {
    if (scores.length === 0) {
      return { mean: 0, std: 0, min: 0, max: 0, count: 0 };
    }
    
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    
    return { mean, std, min, max, count: scores.length };
  }

  /**
   * Generates a classification report with accuracy, precision, recall, and F1 score
   * 
   * @param actual - Array of actual class labels
   * @param predicted - Array of predicted class labels
   * @param probabilities - Array of prediction probabilities (optional)
   * @returns Classification report
   */
  public static classificationReport(
    actual: number[], 
    predicted: number[], 
    probabilities?: number[]
  ): ModelMetrics {
    const numClasses = Math.max(...actual, ...predicted) + 1;
    
    if (numClasses === 2) {
      return this.binaryClassificationReport(actual, predicted, probabilities);
    } else {
      return this.multiclassClassificationReport(actual, predicted, probabilities ? [probabilities] : undefined);
    }
  }

  /**
   * Generates a binary classification report
   * 
   * @param actual - Array of actual class labels (0 or 1)
   * @param predicted - Array of predicted class labels (0 or 1)
   * @param probabilities - Array of prediction probabilities (optional)
   * @returns Binary classification report
   */
  public static binaryClassificationReport(
    actual: number[], 
    predicted: number[], 
    probabilities?: number[]
  ): ModelMetrics {
    const accuracy = this.accuracy(predicted, actual);
    const confusionMatrix = this.confusionMatrix(actual, predicted, 2);
    const precision = this.precision(predicted, actual, 2);
    const recall = this.recall(predicted, actual, 2);
    const f1Score = this.f1Score(predicted, actual, 2);
    
    const metrics: ModelMetrics = {
      taskType: 'classification',
      accuracy,
      confusionMatrix,
      precision,
      recall,
      f1Score,
    };
    
    // Calculate ROC AUC if probabilities are provided
    if (probabilities) {
      metrics.rocAuc = this.rocAuc(probabilities, actual);
    }
    
    return metrics;
  }

  /**
   * Generates a multi-class classification report
   * 
   * @param actual - Array of actual class labels
   * @param predicted - Array of predicted class labels
   * @param probabilities - Array of probability arrays (optional)
   * @returns Multi-class classification report
   */
  public static multiclassClassificationReport(
    actual: number[], 
    predicted: number[], 
    probabilities?: number[][]
  ): ModelMetrics {
    const numClasses = Math.max(...actual, ...predicted) + 1;
    const accuracy = this.accuracy(predicted, actual);
    const confusionMatrix = this.confusionMatrix(actual, predicted, numClasses);
    const precision = this.precision(predicted, actual, numClasses);
    const recall = this.recall(predicted, actual, numClasses);
    const f1Score = this.f1Score(predicted, actual, numClasses);
    
    const metrics: ModelMetrics = {
      taskType: 'classification',
      accuracy,
      confusionMatrix,
      precision,
      recall,
      f1Score,
    };
    
    // Calculate macro ROC AUC if probabilities are provided
    if (probabilities) {
      metrics.rocAuc = this.macroRocAuc(probabilities, actual);
    }
    
    return metrics;
  }

  /**
   * Performs statistical significance test between two sets of scores
   * 
   * @param scores1 - First set of scores
   * @param scores2 - Second set of scores
   * @param alpha - Significance level (default: 0.05)
   * @returns Test result with p-value and significance
   * 
   * @example
   * ```typescript
   * const result = Metrics.significanceTest([0.85, 0.87, 0.83], [0.82, 0.84, 0.80]);
   * console.log('Significant difference:', result.isSignificant);
   * ```
   */
  public static significanceTest(
    scores1: number[],
    scores2: number[],
    alpha: number = 0.05
  ): {
    isSignificant: boolean;
    pValue: number;
    tStatistic: number;
    meanDifference: number;
  } {
    if (scores1.length !== scores2.length) {
      throw new Error('Score arrays must have the same length');
    }
    
    const n = scores1.length;
    if (n < 2) {
      throw new Error('Need at least 2 samples for significance test');
    }
    
    // Calculate differences
    const differences = scores1.map((score, idx) => score - scores2[idx]!);
    const meanDiff = differences.reduce((sum, diff) => sum + diff, 0) / n;
    
    // Calculate standard error
    const variance = differences.reduce((sum, diff) => sum + Math.pow(diff - meanDiff, 2), 0) / (n - 1);
    const standardError = Math.sqrt(variance / n);
    
    // Calculate t-statistic
    const tStatistic = meanDiff / standardError;
    
    // Calculate p-value (two-tailed test)
    // This is a simplified approximation
    const pValue = 2 * (1 - this.studentTCdf(Math.abs(tStatistic), n - 1));
    
    return {
      isSignificant: pValue < alpha,
      pValue,
      tStatistic,
      meanDifference: meanDiff,
    };
  }

  /**
   * Approximates the Student's t-distribution CDF
   * 
   * @private
   * @param t - t-statistic
   * @param df - degrees of freedom
   * @returns Cumulative probability
   */
  private static studentTCdf(t: number, df: number): number {
    // Simplified approximation using normal distribution for large df
    if (df > 30) {
      return this.normalCdf(t);
    }
    
    // For small df, use a rough approximation
    const x = t / Math.sqrt(df);
    return 0.5 + 0.5 * Math.sign(x) * Math.sqrt(1 - Math.exp(-2 * x * x / Math.PI));
  }

  /**
   * Approximates the standard normal CDF
   * 
   * @private
   * @param z - z-score
   * @returns Cumulative probability
   */
  private static normalCdf(z: number): number {
    // Using the error function approximation
    const a = 0.147;
    const sign = z >= 0 ? 1 : -1;
    const absZ = Math.abs(z);
    
    const erf = sign * Math.sqrt(1 - Math.exp(-absZ * absZ * (4 / Math.PI + a * absZ * absZ) / (1 + a * absZ * absZ)));
    return 0.5 + 0.5 * erf;
  }

  /**
   * Calculates precision from confusion matrix
   * 
   * @param confusionMatrix - Confusion matrix
   * @returns Precision per class
   */
  public static precisionFromMatrix(confusionMatrix: number[][]): number[] {
    const precision: number[] = [];
    
    for (let i = 0; i < confusionMatrix.length; i++) {
      const tp = confusionMatrix[i]![i]!;
      const fp = confusionMatrix.map(row => row[i]!).reduce((a, b) => a + b, 0) - tp;
      precision.push(tp + fp > 0 ? tp / (tp + fp) : 0);
    }
    
    return precision;
  }

  /**
   * Calculates recall from confusion matrix
   * 
   * @param confusionMatrix - Confusion matrix
   * @returns Recall per class
   */
  public static recallFromMatrix(confusionMatrix: number[][]): number[] {
    const recall: number[] = [];
    
    for (let i = 0; i < confusionMatrix.length; i++) {
      const tp = confusionMatrix[i]![i]!;
      const fn = confusionMatrix[i]!.reduce((a, b) => a + b, 0) - tp;
      recall.push(tp + fn > 0 ? tp / (tp + fn) : 0);
    }
    
    return recall;
  }

  /**
   * Calculates F1 score from confusion matrix
   * 
   * @param confusionMatrix - Confusion matrix
   * @returns F1 score per class
   */
  public static f1ScoreFromMatrix(confusionMatrix: number[][]): number[] {
    const precision = this.precisionFromMatrix(confusionMatrix);
    const recall = this.recallFromMatrix(confusionMatrix);
    
    return precision.map((prec, idx) => {
      const rec = recall[idx]!;
      return prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
    });
  }

  /**
   * Calculates Mean Squared Error (MSE) for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns MSE value
   * 
   * @example
   * ```typescript
   * const mse = Metrics.meanSquaredError([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('MSE:', mse); // 0.09
   * ```
   */
  public static meanSquaredError(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    const sumSquaredErrors = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.pow(pred - actualVal, 2) : 0);
    }, 0);
    
    return sumSquaredErrors / predicted.length;
  }

  /**
   * Calculates Root Mean Squared Error (RMSE) for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns RMSE value
   * 
   * @example
   * ```typescript
   * const rmse = Metrics.rootMeanSquaredError([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('RMSE:', rmse); // 0.3
   * ```
   */
  public static rootMeanSquaredError(predicted: number[], actual: number[]): number {
    return Math.sqrt(this.meanSquaredError(predicted, actual));
  }

  /**
   * Calculates Mean Absolute Error (MAE) for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns MAE value
   * 
   * @example
   * ```typescript
   * const mae = Metrics.meanAbsoluteError([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('MAE:', mae); // 0.23
   * ```
   */
  public static meanAbsoluteError(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    const sumAbsoluteErrors = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.abs(pred - actualVal) : 0);
    }, 0);
    
    return sumAbsoluteErrors / predicted.length;
  }

  /**
   * Calculates R-squared (coefficient of determination) for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns R² value
   * 
   * @example
   * ```typescript
   * const r2 = Metrics.r2Score([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('R²:', r2); // 0.85
   * ```
   */
  public static r2Score(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    // Calculate mean of actual values
    const validActual = actual.filter(val => val !== undefined);
    if (validActual.length === 0) {
      return 0;
    }
    
    const actualMean = validActual.reduce((sum, val) => sum + val, 0) / validActual.length;
    
    // Calculate sum of squared residuals
    const ssRes = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      return sum + (actualVal !== undefined ? Math.pow(actualVal - pred, 2) : 0);
    }, 0);
    
    // Calculate total sum of squares
    const ssTot = validActual.reduce((sum, val) => sum + Math.pow(val - actualMean, 2), 0);
    
    // Return R²
    return ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
  }

  /**
   * Calculates Mean Absolute Percentage Error (MAPE) for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns MAPE value as percentage
   * 
   * @example
   * ```typescript
   * const mape = Metrics.meanAbsolutePercentageError([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('MAPE:', mape); // 12.5%
   * ```
   */
  public static meanAbsolutePercentageError(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    const sumPercentageErrors = predicted.reduce((sum, pred, idx) => {
      const actualVal = actual[idx];
      if (actualVal !== undefined && actualVal !== 0) {
        return sum + Math.abs((actualVal - pred) / actualVal);
      }
      return sum;
    }, 0);
    
    return (sumPercentageErrors / predicted.length) * 100;
  }

  /**
   * Calculates Median Absolute Error for regression
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns Median absolute error
   * 
   * @example
   * ```typescript
   * const medianAe = Metrics.medianAbsoluteError([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('Median AE:', medianAe); // 0.3
   * ```
   */
  public static medianAbsoluteError(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) {
      throw new Error('Predicted and actual arrays must have the same length');
    }
    
    if (predicted.length === 0) {
      return 0;
    }
    
    const absoluteErrors = predicted.map((pred, idx) => {
      const actualVal = actual[idx];
      return actualVal !== undefined ? Math.abs(pred - actualVal) : 0;
    }).sort((a, b) => a - b);
    
    const midIndex = Math.floor(absoluteErrors.length / 2);
    return absoluteErrors.length > 0 ? (absoluteErrors[midIndex] || 0) : 0;
  }

  /**
   * Calculates comprehensive regression metrics
   * 
   * @param predicted - Array of predicted values
   * @param actual - Array of actual values
   * @returns Complete regression metrics
   * 
   * @example
   * ```typescript
   * const metrics = Metrics.calculateRegressionMetrics([2.5, 3.2, 1.8], [2.0, 3.5, 2.1]);
   * console.log('MSE:', metrics.mse);
   * console.log('RMSE:', metrics.rmse);
   * console.log('MAE:', metrics.mae);
   * console.log('R²:', metrics.r2);
   * ```
   */
  public static calculateRegressionMetrics(predicted: number[], actual: number[]): RegressionMetrics {
    return {
      mse: this.meanSquaredError(predicted, actual),
      rmse: this.rootMeanSquaredError(predicted, actual),
      mae: this.meanAbsoluteError(predicted, actual),
      r2: this.r2Score(predicted, actual),
      mape: this.meanAbsolutePercentageError(predicted, actual),
      medianAe: this.medianAbsoluteError(predicted, actual),
      n: predicted.length,
    };
  }

  /**
   * Calculates comprehensive model metrics for both classification and regression
   * 
   * @param predicted - Array of predicted values (class indices for classification, numeric values for regression)
   * @param actual - Array of actual values (class indices for classification, numeric values for regression)
   * @param taskType - Type of task ('classification' or 'regression')
   * @param probabilities - Array of probability arrays for classification (optional)
   * @returns Complete model metrics
   * 
   * @example
   * ```typescript
   * // For classification
   * const classMetrics = Metrics.calculateModelMetrics([0, 1, 1], [0, 1, 0], 'classification');
   * 
   * // For regression
   * const regMetrics = Metrics.calculateModelMetrics([2.5, 3.2, 1.8], [2.0, 3.5, 2.1], 'regression');
   * ```
   */
  public static calculateModelMetrics(
    predicted: number[],
    actual: number[],
    taskType: 'classification' | 'regression',
    probabilities?: number[][]
  ): ModelMetrics {
    const baseMetrics: ModelMetrics = {
      taskType,
    };

    if (taskType === 'classification') {
      // Calculate classification metrics
      const classificationMetrics = this.calculateMetrics(predicted, actual, probabilities);
      
      return {
        ...baseMetrics,
        ...classificationMetrics,
      };
    } else {
      // Calculate regression metrics
      const regressionMetrics = this.calculateRegressionMetrics(predicted, actual);
      
      return {
        ...baseMetrics,
        mse: regressionMetrics.mse,
        rmse: regressionMetrics.rmse,
        mae: regressionMetrics.mae,
        r2: regressionMetrics.r2,
        mape: regressionMetrics.mape,
        medianAe: regressionMetrics.medianAe,
      };
    }
  }
} 