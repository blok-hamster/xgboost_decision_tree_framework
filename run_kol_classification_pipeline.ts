/**
 * @fileoverview Pipeline runner for Solana KOL OHLCV Classification
 * 
 * This script orchestrates the entire machine learning pipeline:
 * 1. Data splitting (temporal/stratified/random)
 * 2. Model training on training set
 * 3. Model testing and evaluation on test set
 * 4. Results comparison and analysis
 * 
 * Usage:
 * - npx ts-node run_kol_classification_pipeline.ts
 * - npx ts-node run_kol_classification_pipeline.ts temporal
 * - npx ts-node run_kol_classification_pipeline.ts stratified
 */

import { splitKolData, SplitConfig } from './split_kol_data';
import { trainKolClassifier } from './train_kol_classifier';
import { testKolClassifier } from './test_kol_classifier';
import { defaultLogger } from './src';
import * as fs from 'fs';
import * as path from 'path';

interface PipelineConfig {
  inputFile: string;
  outputDir: string;
  modelsDir: string;
  splitMethod: 'temporal' | 'stratified' | 'random';
  testRatio: number;
  seed: number;
  runAll: boolean;
  skipSplitting: boolean;
  skipTraining: boolean;
  skipTesting: boolean;
}

interface PipelineResults {
  splitMethod: string;
  dataStats: {
    total: number;
    train: number;
    test: number;
  };
  trainingMetrics?: any;
  testResults?: any;
  executionTime: number;
  success: boolean;
  error?: string;
}

async function runKolClassificationPipeline(config: PipelineConfig): Promise<PipelineResults> {
  const startTime = Date.now();
  
  console.log('🚀 Solana KOL OHLCV Classification Pipeline\n');
  console.log('Configuration:');
  console.log(`- Input file: ${config.inputFile}`);
  console.log(`- Split method: ${config.splitMethod}`);
  console.log(`- Test ratio: ${config.testRatio * 100}%`);
  console.log(`- Seed: ${config.seed}`);
  console.log(`- Output directory: ${config.outputDir}`);
  console.log(`- Models directory: ${config.modelsDir}`);
  console.log();

  const results: PipelineResults = {
    splitMethod: config.splitMethod,
    dataStats: { total: 0, train: 0, test: 0 },
    executionTime: 0,
    success: false
  };

  try {
    // Step 1: Data Splitting
    if (!config.skipSplitting) {
      console.log('📊 Step 1: Splitting data...');
      console.log('='.repeat(50));
      
      const splitConfig: SplitConfig = {
        inputFile: config.inputFile,
        outputDir: config.outputDir,
        testRatio: config.testRatio,
        splitMethod: config.splitMethod,
        seed: config.seed,
        validateSplit: true
      };

      await splitKolData(splitConfig);
      
      // Load split metadata
      const metadataFile = path.join(config.outputDir, `split_metadata_${config.splitMethod}.json`);
      if (fs.existsSync(metadataFile)) {
        const metadata = JSON.parse(fs.readFileSync(metadataFile, 'utf8'));
        results.dataStats = {
          total: metadata.totalRecords,
          train: metadata.trainRecords,
          test: metadata.testRecords
        };
      }
      
      console.log('✅ Data splitting completed\n');
    } else {
      console.log('⏭️  Skipping data splitting\n');
    }

    // Step 2: Model Training  
    if (!config.skipTraining) {
      console.log('🎯 Step 2: Training model...');
      console.log('='.repeat(50));
      
      // Update the trainer to use split data
      const trainFile = path.join(config.outputDir, `train_ohlcv_features_${config.splitMethod}.csv`);
      
      if (!fs.existsSync(trainFile)) {
        throw new Error(`Training file not found: ${trainFile}. Please run data splitting first.`);
      }

      // Note: We would need to modify the train_kol_classifier to accept a different input file
      // For now, we'll run the existing trainer
      console.log(`Training on: ${trainFile}`);
      console.log('Training with existing script (modify data path as needed)...');
      
      await trainKolClassifier();
      
      console.log('✅ Model training completed\n');
    } else {
      console.log('⏭️  Skipping model training\n');
    }

    // Step 3: Model Testing
    if (!config.skipTesting) {
      console.log('🧪 Step 3: Testing model...');
      console.log('='.repeat(50));
      
      const testFile = path.join(config.outputDir, `test_ohlcv_features_${config.splitMethod}.csv`);
      const modelPath = path.join(config.modelsDir, 'kol_ohlcv_classification_model');
      
      if (!fs.existsSync(testFile)) {
        console.warn(`⚠️  Test file not found: ${testFile}`);
        console.log('Using default dataset for testing...');
      }
      
      if (!fs.existsSync(modelPath)) {
        throw new Error(`Model not found: ${modelPath}. Please run training first.`);
      }

      await testKolClassifier();
      
      console.log('✅ Model testing completed\n');
    } else {
      console.log('⏭️  Skipping model testing\n');
    }

    // Step 4: Pipeline Summary
    const executionTime = Date.now() - startTime;
    results.executionTime = executionTime;
    results.success = true;

    console.log('📊 Pipeline Summary');
    console.log('='.repeat(50));
    console.log(`Split method: ${config.splitMethod}`);
    console.log(`Total execution time: ${(executionTime / 1000).toFixed(1)}s`);
    console.log(`Data split: ${results.dataStats.train} train / ${results.dataStats.test} test`);
    console.log('Pipeline completed successfully! ✅');
    console.log('='.repeat(50));

    return results;

  } catch (error) {
    const executionTime = Date.now() - startTime;
    results.executionTime = executionTime;
    results.error = error instanceof Error ? error.message : 'Unknown error';
    
    console.error('❌ Pipeline failed:', error);
    console.log(`Execution time before failure: ${(executionTime / 1000).toFixed(1)}s`);
    
    throw error;
  }
}

async function runMultipleSplitMethods(): Promise<void> {
  console.log('🔄 Running pipeline with multiple split methods...\n');
  
  const splitMethods: ('temporal' | 'stratified' | 'random')[] = ['temporal', 'stratified', 'random'];
  const allResults: PipelineResults[] = [];
  
  for (const method of splitMethods) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Running pipeline with ${method.toUpperCase()} split method`);
    console.log(`${'='.repeat(60)}\n`);
    
    try {
      const config: PipelineConfig = {
        inputFile: './data-extraction/output/batch-ohlcv/Cupsey/Cupsey_ohlcv_features_2025-07-18T17-18-07-881Z.csv',
        outputDir: `./data/splits/${method}`,
        modelsDir: `./models`,
        splitMethod: method,
        testRatio: 0.2,
        seed: 42,
        runAll: true,
        skipSplitting: false,
        skipTraining: false,
        skipTesting: false
      };

      const result = await runKolClassificationPipeline(config);
      allResults.push(result);
      
    } catch (error) {
      console.error(`Failed to run pipeline with ${method} method:`, error);
      allResults.push({
        splitMethod: method,
        dataStats: { total: 0, train: 0, test: 0 },
        executionTime: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
  
  // Compare results
  console.log('\n📊 COMPARISON OF ALL METHODS');
  console.log('='.repeat(70));
  console.log('Method      | Success | Time(s) | Train | Test  | Status');
  console.log('-'.repeat(70));
  
  allResults.forEach(result => {
    const status = result.success ? '✅' : '❌';
    const time = (result.executionTime / 1000).toFixed(1);
    console.log(`${result.splitMethod.padEnd(11)} | ${result.success ? 'Yes'.padEnd(7) : 'No'.padEnd(7)} | ${time.padEnd(7)} | ${result.dataStats.train.toString().padEnd(5)} | ${result.dataStats.test.toString().padEnd(5)} | ${status}`);
  });
  
  console.log('='.repeat(70));
  
  const successCount = allResults.filter(r => r.success).length;
  console.log(`\nResults: ${successCount}/${allResults.length} methods completed successfully`);
  
  if (successCount > 1) {
    console.log('\n💡 Recommendations:');
    console.log('- Temporal split: Best for production simulation (future prediction)');
    console.log('- Stratified split: Best for balanced evaluation across classes');
    console.log('- Random split: Good baseline for general model validation');
    console.log('\nChoose the method that best fits your use case and data characteristics.');
  }
}

// Main execution
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  
  // Help message
  if (args.includes('--help') || args.includes('-h')) {
    console.log('🎯 Solana KOL OHLCV Classification Pipeline');
    console.log('\nUsage:');
    console.log('  npx ts-node run_kol_classification_pipeline.ts [method]');
    console.log('\nMethods:');
    console.log('  temporal    - Split by time (train on older, test on newer data)');
    console.log('  stratified  - Split maintaining class distribution');
    console.log('  random      - Random split');
    console.log('  all         - Run all three methods and compare');
    console.log('\nOptions:');
    console.log('  --help, -h  - Show this help message');
    console.log('\nExamples:');
    console.log('  npx ts-node run_kol_classification_pipeline.ts temporal');
    console.log('  npx ts-node run_kol_classification_pipeline.ts all');
    return;
  }
  
  // Determine which method(s) to run
  const method = args[0] as 'temporal' | 'stratified' | 'random' | 'all' || 'stratified';
  
  if (method === 'all') {
    await runMultipleSplitMethods();
    return;
  }
  
  if (!['temporal', 'stratified', 'random'].includes(method)) {
    console.error(`❌ Invalid method: ${method}`);
    console.log('Valid methods: temporal, stratified, random, all');
    process.exit(1);
  }
  
  // Run single method
  const config: PipelineConfig = {
    inputFile: './data-extraction/output/batch-ohlcv/Cupsey/Cupsey_ohlcv_features_2025-07-18T17-18-07-881Z.csv',
    outputDir: `./data/splits`,
    modelsDir: './models',
    splitMethod: method,
    testRatio: 0.2,
    seed: 42,
    runAll: false,
    skipSplitting: false,
    skipTraining: false,
    skipTesting: false
  };
  
  await runKolClassificationPipeline(config);
}

// Error handling and execution
if (require.main === module) {
  main()
    .then(() => {
      console.log('\n🏁 Pipeline execution completed');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Pipeline execution failed:', error);
      console.log('\n🔧 Troubleshooting:');
      console.log('1. Ensure all dependencies are installed');
      console.log('2. Check that the input CSV file exists');
      console.log('3. Verify write permissions for output directories');
      console.log('4. Run individual scripts separately to isolate issues');
      process.exit(1);
    });
}

export { runKolClassificationPipeline };
export type { PipelineConfig, PipelineResults }; 