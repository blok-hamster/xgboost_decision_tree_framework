/**
 * @fileoverview Quick start example for the Decision Tree Model Toolkit
 * 
 * This example demonstrates the complete end-to-end workflow:
 * 1. Data preparation and analysis
 * 2. Model training with cross-validation
 * 3. Model persistence and loading
 * 4. Inference on new data
 */

import {
  Trainer,
  Model,
  FeatureAnalyzer,
  Metrics,
  Utils,
  createTestDataset,
  defaultLogger,
} from '../src';

async function quickStartExample(): Promise<void> {
  console.log('🚀 Decision Tree Model Toolkit - Quick Start Example\n');

  // Step 1: Generate test data
  console.log('📊 Step 1: Generating test data...');
  const data = createTestDataset(6000, 'binary');
  console.log(`Generated ${data.length} records`);
  console.log('Sample record:', data[0]);
  console.log();

  // Step 2: Analyze features
  console.log('🔍 Step 2: Analyzing features...');
  const analyzer = new FeatureAnalyzer(defaultLogger);
  const analysis = analyzer.analyzeFeatures(
    data,
    ['category', 'color', 'size'],  // categorical features
    ['value', 'rating'],            // numerical features
    'target'                        // target variable
  );
  
  console.log('Feature analysis results:');
  console.log('- Categorical specs:', analysis.categoricalSpecs);
  console.log('- Dataset stats:', analysis.datasetStats);
  console.log();

  // Step 3: Train model
  console.log('🎯 Step 3: Training model...');
  const trainer = new Trainer({
    categoricalFeatures: ['category', 'color', 'size'],
    numericFeatures: ['value', 'rating'],
    target: 'target',
    xgbParams: {
      objective: 'binary:logistic',
      max_depth: 9,
      eta: 0.1,
      num_boost_round: 100,
    },
  }, defaultLogger);

  // Load data
  await trainer.loadRaw(data);
  
  // Train with cross-validation
  await trainer.train({
    useCrossValidation: true,
    nFolds: 5,
    testRatio: 0.2,
    shuffle: true,
    seed: 42,
  });

  // Get training metrics
  const metrics = trainer.evaluate();
  console.log('Training metrics:');
  console.log(`- Accuracy: ${(metrics.accuracy! * 100).toFixed(2)}%`);
  console.log(`- Precision: [${metrics.precision!.map(p => (p * 100).toFixed(1)).join(', ')}]%`);
  console.log(`- Recall: [${metrics.recall!.map(r => (r * 100).toFixed(1)).join(', ')}]%`);
  console.log(`- F1-Score: [${metrics.f1Score!.map(f => (f * 100).toFixed(1)).join(', ')}]%`);
  if (metrics.crossValidationScores) {
    const cvStats = Metrics.crossValidationStats(metrics.crossValidationScores);
    console.log(`- CV Mean: ${(cvStats.mean * 100).toFixed(2)}% ± ${(cvStats.std * 100).toFixed(2)}%`);
  }
  console.log();

  // Step 4: Save model
  console.log('💾 Step 4: Saving model...');
  const modelPath = './models/quick_start_model';
  await trainer.save(modelPath);
  console.log(`Model saved to: ${modelPath}`);
  console.log();

  // Step 5: Load model and make predictions
  console.log('🔮 Step 5: Loading model and making predictions...');
  const model = await Model.load(modelPath, defaultLogger);
  
  // Test predictions on new data
  const testSamples = [
    { category: 'A', color: 'red', size: 'large', value: 75.5, rating: 4.2 },
    { category: 'B', color: 'blue', size: 'medium', value: 45.8, rating: 3.1 },
    { category: 'C', color: 'green', size: 'small', value: 23.7, rating: 2.8 },
    { category: 'D', color: 'yellow', size: 'large', value: 89.2, rating: 4.8 },
  ];

  console.log('Making predictions:');
  for (const sample of testSamples) {
    const result = model.predictWithProbabilities(sample);
    console.log(`Sample: ${JSON.stringify(sample)}`);
    console.log(`  → Predicted: ${result.classLabel} (confidence: ${(result.probability! * 100).toFixed(1)}%)`);
    console.log(`  → Probabilities: [${result.probabilities!.map(p => (p * 100).toFixed(1)).join(', ')}]%`);
  }
  console.log();

  // Step 6: Batch prediction
  console.log('📦 Step 6: Batch prediction...');
  const batchResult = model.predictBatchWithProbabilities(testSamples);
  console.log(`Batch predictions completed for ${batchResult.predictions.length} samples`);
  console.log(`Average confidence: ${(batchResult.averageConfidence! * 100).toFixed(1)}%`);
  console.log();

  // Step 7: Model information
  console.log('ℹ️ Step 7: Model information...');
  const metadata = model.getMetadata();
  console.log(`Model version: ${metadata.version}`);
  console.log(`Created at: ${metadata.createdAt}`);
  console.log(`Classes: [${metadata.classes!.join(', ')}]`);
  console.log(`Features: ${model.getFeatureCount()} total`);
  console.log(`- Categorical: [${metadata.categoricalFeatures.join(', ')}]`);
  console.log(`- Numerical: [${metadata.numericFeatures.join(', ')}]`);
  console.log(`One-vs-Rest: ${metadata.isOneVsRest}`);
  console.log();

  // Step 8: Feature transformation example
  console.log('🔧 Step 8: Feature transformation example...');
  const sampleFeatures = model.transform(testSamples[0]!);
  console.log(`Original: ${JSON.stringify(testSamples[0])}`);
  console.log(`Transformed: [${sampleFeatures.slice(0, 10).map(f => f.toFixed(2)).join(', ')}...] (${sampleFeatures.length} features)`);
  console.log();

  // Step 9: Validation example
  console.log('✅ Step 9: Data validation example...');
  const validationResult = model.validateRecord({ category: 'A', color: 'red' }); // Missing features
  console.log('Validation result:', validationResult);
  console.log();

  console.log('🎉 Quick start example completed successfully!');
  console.log('\nNext steps:');
  console.log('- Try with your own data using trainer.loadCSV() or trainer.loadJSON()');
  console.log('- Experiment with different XGBoost parameters');
  console.log('- Use cross-validation to optimize hyperparameters');
  console.log('- Deploy the model in production using model.predict()');
}

// Performance measurement example
async function performanceExample(): Promise<void> {
  console.log('\n⚡ Performance Measurement Example\n');

  // Generate larger dataset
  const data = createTestDataset(10000, 'binary');
  
  // Measure training time
  const { duration: trainingTime, result: trainer } = Utils.measureTime(() => {
    return new Trainer({
      categoricalFeatures: ['category', 'color', 'size'],
      numericFeatures: ['value', 'rating'],
      target: 'target',
      xgbParams: {
        objective: 'binary:logistic',
        max_depth: 4,
        eta: 0.3,
        num_boost_round: 20,
      },
    });
  });

  console.log(`Trainer initialization: ${trainingTime.toFixed(2)}ms`);

  await trainer.loadRaw(data);
  
  const { duration: modelTrainingTime } = await Utils.measureTimeAsync(async () => {
    // Train with cross-validation
    await trainer.train({
      useCrossValidation: true,
      nFolds: 5,
      testRatio: 0.2,
      shuffle: true,
      seed: 42,
    });
  });

  console.log(`Model training: ${modelTrainingTime.toFixed(2)}ms`);
  
  // Save and measure loading time
  await trainer.save('./models/performance_test');
  
  const start = performance.now();
  const model = await Model.load('./models/performance_test');
  const loadingTime = performance.now() - start;

  console.log(`Model loading: ${loadingTime.toFixed(2)}ms`);

  // Measure prediction time
  const testSample = { category: 'A', color: 'red', size: 'large', value: 75.5, rating: 4.2 };
  
  const { duration: predictionTime } = Utils.measureTime(() => {
    return model.predict(testSample);
  });

  console.log(`Single prediction: ${predictionTime.toFixed(3)}ms`);
  
  // Batch prediction performance
  const batchSamples = Array(100).fill(testSample);
  const { duration: batchPredictionTime } = Utils.measureTime(() => {
    return model.predictBatch(batchSamples);
  });

  console.log(`Batch prediction (100 samples): ${batchPredictionTime.toFixed(2)}ms`);
  console.log(`Average per sample: ${(batchPredictionTime / 100).toFixed(3)}ms`);

  // Performance summary
  console.log('\n📊 Performance Summary:');
  console.log(`- Data size: ${data.length} records`);
  console.log(`- Training: ${modelTrainingTime.toFixed(0)}ms`);
  console.log(`- Loading: ${loadingTime.toFixed(0)}ms`);
  console.log(`- Inference: ${predictionTime.toFixed(3)}ms per sample`);
  
  if (predictionTime < 50) {
    console.log('✅ Performance target met: <50ms per sample');
  } else {
    console.log('⚠️ Performance target not met: >50ms per sample');
  }
}

// Run examples
if (require.main === module) {
  //quickStartExample()
  performanceExample()
    .then(() => {
      console.log('Performance example completed successfully!');
    })
    .catch(error => {
      console.error('❌ Example failed:', error);
      process.exit(1);
    });
}

export { quickStartExample, performanceExample }; 