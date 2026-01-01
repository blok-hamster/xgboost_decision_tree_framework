
import path from 'path';
import fs from 'fs';
import { Trainer, defaultLogger } from '../../src';
import { CONFIG } from './config';

async function trainAdvancedModel() {
  console.log("🚀 Starting Advanced Model Training...");

  // 1. Define the Input/Output Path
  // 1. Define the Input/Output Path
  const dataPath = CONFIG.OUTPUT_FILE; 
  
  // Generate unique model version/timestamp
  // Auto-increment version in package.json
  const packageJsonPath = path.resolve(__dirname, '../../package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  const versionParts = packageJson.version.split('.').map(Number);
  versionParts[2]++; // Increment patch
  const newVersion = versionParts.join('.');
  packageJson.version = newVersion;
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  console.log(`🆙 Incremented Package Version: ${newVersion}`);

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const version = newVersion;
  const modelDirName = `model_v${version}_${timestamp}`;
  const modelSavePath = path.resolve(__dirname, `../model_output/${modelDirName}`);

  // 2. Configure the Trainer
  const trainer = new Trainer({
    // Target Variable
    target: 'label',
    taskType: 'classification',

    // Feature Definitions (The 4 Pillars)
    categoricalFeatures: [],
    numericFeatures: [
      'rvol',       // Volume Anomaly
      'natr',       // Volatility
      'vwap_dev',   // Mean Reversion
      'stoch_k',    // Momentum (Fast)
      'stoch_d',    // Momentum (Slow)
      'upper_wick', // Rejection (Bearish)
      'lower_wick', // Rejection (Bullish)
      'market_cap',  // Scale (Absolute Value)
      'token_age_hours' // Time in market (Proxy)
    ],

    // XGBoost Hyperparameters
    xgbParams: {
      objective: 'binary:logistic',
      eval_metric: 'logloss',
      max_depth: 6,
      eta: 0.05,
      subsample: 0.8,
      colsample_bytree: 0.8
    },
  }, defaultLogger);

  // 3. Run Training
  try {
    console.log(`📂 Loading data from: ${dataPath}`);
    
    // 3. Load Data & Balance Classes
    console.log(`📂 Loading data from: ${dataPath}`);

    // Load full dataset first
    const rawData = await trainer.loadCSV(dataPath, {
        parseNumbers: true,
        skipEmptyLines: true,
        headers: true
    });
    
    // Manual Undersampling logic (since Trainer doesn't natively support it yet)
    // We access the raw loaded data from the trainer instance if possible, 
    // OR we can just use the library's ability to handle this if it exists.
    // Given the previous viewing of the library, it seems to just wrap XGBoost.
    // Plan B: We can just use the 'weight' column if the trainer supports it,
    // but undersampling is stronger for this specific "lazy model" issue.
    
    // Because `loadCSV` internal state is hard to modify from outside without accessors,
    // We will re-implement a simple loader here, balance, and save a temp file.
    
    const fs = require('fs');
    const { parse } = require('csv-parse/sync');
    const { stringify } = require('csv-stringify/sync');
    
    // Create directory early
    if (!fs.existsSync(modelSavePath)) {
        fs.mkdirSync(modelSavePath, { recursive: true });
    }

    const fileContent = fs.readFileSync(dataPath, 'utf-8');
    const records = parse(fileContent, { columns: true, cast: true });
    
    // 3a. Create Holdout Split (80/20) - BEFORE Balancing
    // Shuffle raw data first
    const shuffledRaw = records.sort(() => 0.5 - Math.random());
    const splitIndex = Math.floor(shuffledRaw.length * 0.8);
    const trainSet = shuffledRaw.slice(0, splitIndex);
    const testSet = shuffledRaw.slice(splitIndex);

    console.log(`✂️  Data Split: Train (${trainSet.length}) / Test (${testSet.length})`);
    
    // Save Test Set to Model Folder
    const testSetPath = path.join(modelSavePath, 'test_set.csv');
    fs.writeFileSync(testSetPath, stringify(testSet, { header: true }));
    console.log(`💾 Saved Holdout Test Set to: ${testSetPath}`);

    // 3b. Balance Training Data (Undersample Majority Class)
    const goods = trainSet.filter((r: any) => r.label === 'good');
    const bads = trainSet.filter((r: any) => r.label === 'bad');
    const neutrals = trainSet.filter((r: any) => r.label === 'neutral');

    console.log(`📊 Train Set Dist: Good: ${goods.length}, Bad: ${bads.length}`);
    
    // Undersample Goods to match Bads
    const limit = bads.length;
    // Goods already shuffled from initial split
    const selectedGoods = goods.slice(0, limit);
    
    const balancedTrainData = [...selectedGoods, ...bads, ...neutrals];
    const finalTrainData = balancedTrainData.sort(() => 0.5 - Math.random());
    
    console.log(`⚖️  Balanced Train Set: ${finalTrainData.length} records`);
    
    const tempPath = path.resolve(__dirname, '../training_data_balanced.csv');
    fs.writeFileSync(tempPath, stringify(finalTrainData, { header: true }));
    
    // 4. Train Model
    await trainer.loadCSV(tempPath, {
        parseNumbers: true,
        skipEmptyLines: true,
        headers: true
    });
    
    await trainer.train({
        useCrossValidation: true,
        nFolds: 5,
        shuffle: true,
        seed: 42
    });

    console.log("\n✅ Training Complete!");
    
    await trainer.save(modelSavePath);
    console.log(`💾 Model saved to: ${modelSavePath}`);
    
    // 5. Generate Test Script
    const testScriptContent = `
import { Model, defaultLogger, Metrics } from '../../../src';
import path from 'path';
import fs from 'fs';
import { parse } from 'csv-parse/sync';

async function testModel() {
  console.log("🚀 Testing Model Version: ${version}");
  console.log("📂 Model Path: ${modelDirName}");

  const modelPath = path.join(__dirname);
  const testDataPath = path.join(__dirname, 'test_set.csv');

  // 1. Load Model
  console.log("Loading model...");
  const model = await Model.load(modelPath, defaultLogger);
  const classes = model.getClasses();
  console.log("Target Classes:", classes);

  // 2. Load Test Data
  console.log("Loading test data...");
  const fileContent = fs.readFileSync(testDataPath, 'utf-8');
  const records = parse(fileContent, { columns: true, cast: true });
  console.log(\`Loaded \${records.length} test records.\`);

  // 3. Run Predictions
  console.log("running predictions...");
  const predicted: number[] = [];
  const actual: number[] = [];

  for (const record of records) {
      // Predict
      const result = model.predictWithProbabilities(record);
      predicted.push(result.classIndex!);

      // Get Actual
      const actualLabel = record.label;
      const actualIndex = classes.indexOf(actualLabel);
      
      if (actualIndex === -1) {
          console.warn(\`Unknown label in test data: \${actualLabel}\`);
          continue;
      }
      actual.push(actualIndex);
  }

  // 4. Calculate Metrics
  const metrics = Metrics.calculateMetrics(predicted, actual);

  console.log("\\n📊 Holdout Test Results:");
  console.log(\`Accuracy:  \${((metrics.accuracy || 0) * 100).toFixed(2)}%\`);
  
  const precision = metrics.precision || [0, 0];
  const recall = metrics.recall || [0, 0];
  const f1Score = metrics.f1Score || [0, 0];

  console.log(\`Precision: \${(precision[1] * 100).toFixed(2)}% (Good Trades)\`);
  console.log(\`Recall:    \${(recall[0] * 100).toFixed(2)}% (Bad Trades)\`);
  console.log(\`F1-Score:  \${(f1Score[1]).toFixed(2)}\`);
}

testModel();
`;
    const scriptPath = path.join(modelSavePath, 'test_model.ts');
    fs.writeFileSync(scriptPath, testScriptContent);
    console.log(`📝 Generated Test Script: ${scriptPath}`);
    console.log(`👉 Run it with: npx ts-node --transpile-only ${scriptPath}`);

    // 6. Bundle Feature Extractor (Standalone)
    const featureExtractorSource = fs.readFileSync(path.resolve(__dirname, '../../src/core/FeatureExtractor.ts'), 'utf-8');
    
    // Inline the types to make it truly portable
    const standaloneExtractor = `
/**
 * OHLCV Candle structure
 */
export interface Candle {
  timestamp: number; // Seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  market_cap?: number; // Estimated market cap
}

/**
 * Advanced features calculated from OHLCV data
 */
export interface AdvancedFeatures {
  rvol: number;
  k_percentage: number;
  d_percentage: number;
  natr: number;
  vwap_deviation: number;
  upper_wick_ratio: number;
  lower_wick_ratio: number;
  market_cap: number;
  visible_token_age: number; // Hours of data available before buy
}

${featureExtractorSource.replace("import { Candle, AdvancedFeatures } from '../types';", "// Types inlined above")}
`;

    const extractorPath = path.join(modelSavePath, 'feature_extractor.ts');
    fs.writeFileSync(extractorPath, standaloneExtractor);
    console.log(`📦 Bundled Feature Extractor: ${extractorPath}`);

  } catch (error) {
    console.error("❌ Training Failed:", error);
  }
}

trainAdvancedModel();
