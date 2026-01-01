
import { Model, defaultLogger, Metrics } from '../../../src';
import path from 'path';
import fs from 'fs';
import { parse } from 'csv-parse/sync';

async function testModel() {
  console.log("🚀 Testing Model Version: 1.0.4");
  console.log("📂 Model Path: model_v1.0.4_2026-01-01T09-38-07-977Z");

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
  console.log(`Loaded ${records.length} test records.`);

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
          console.warn(`Unknown label in test data: ${actualLabel}`);
          continue;
      }
      actual.push(actualIndex);
  }

  // 4. Calculate Metrics
  const metrics = Metrics.calculateMetrics(predicted, actual);

  console.log("\n📊 Holdout Test Results:");
  console.log(`Accuracy:  ${((metrics.accuracy || 0) * 100).toFixed(2)}%`);
  
  const precision = metrics.precision || [0, 0];
  const recall = metrics.recall || [0, 0];
  const f1Score = metrics.f1Score || [0, 0];

  console.log(`Precision: ${(precision[1] * 100).toFixed(2)}% (Good Trades)`);
  console.log(`Recall:    ${(recall[0] * 100).toFixed(2)}% (Bad Trades)`);
  console.log(`F1-Score:  ${(f1Score[1]).toFixed(2)}`);
}

testModel();
