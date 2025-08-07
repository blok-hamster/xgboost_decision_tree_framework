/**
 * @fileoverview Testing script for Solana KOL OHLCV Features Classification
 * 
 * This script loads the trained classification model and performs testing
 * on new OHLCV features data to evaluate model performance and make predictions.
 * 
 * Features:
 * - Model loading and validation
 * - Batch prediction on test datasets
 * - Performance evaluation with detailed metrics
 * - Individual prediction examples
 * - Feature importance analysis
 */

import {
  Model,
  DataLoader,
  defaultLogger,
  RawDataRecord,
} from './src';
import * as fs from 'fs';

interface TestSample {
  [key: string]: any; // Add index signature for RawDataRecord compatibility
  candle_count: number;
  market_cap_candle_count: number;
  price_at_buy: number;
  avg_price: number;
  total_volume_usd: number;
  market_cap_at_buy: number;
  avg_market_cap: number;
  last_available_market_cap: number;
  price_change_1h_percent: number;
  price_change_4h_percent: number;
  price_change_24h_percent: number;
  market_cap_change_1h_percent: number;
  market_cap_change_4h_percent: number;
  market_cap_change_24h_percent: number;
  price_volatility_1h: number;
  price_volatility_4h: number;
  price_volatility_24h: number;
  market_cap_volatility_1h: number;
  market_cap_volatility_4h: number;
  market_cap_volatility_24h: number;
  price_atr_24h: number;
  market_cap_atr_24h: number;
  price_range_24h_percent: number;
  market_cap_range_24h_percent: number;
  price_rsi: number;
  market_cap_rsi: number;
  price_macd: number;
  market_cap_macd: number;
  price_sma_20: number;
  market_cap_sma_20: number;
  price_ema_20: number;
  market_cap_ema_20: number;
  price_bollinger_position: number;
  market_cap_bollinger_position: number;
  price_momentum_score: number;
  market_cap_momentum_score: number;
  buy_volume_ratio_24h: number;
  volume_trend_24h: number;
  buy_pressure: number;
  volume_price_correlation: number;
  performance_label?: string; // Optional for new predictions
}

async function testKolClassifier(): Promise<void> {
  console.log('🧪 Solana KOL OHLCV Features Classification Testing\n');

  // Define paths
  const modelPath = './models/cupsey_ohlcv_classification_model';
  const testDataPath = './data-extraction/output/batch-ohlcv/Cupsey/Cupsey_ohlcv_features_2025-07-18T17-18-07-881Z.csv';

  // Check if model exists
  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model not found: ${modelPath}`);
    console.log('Please run the training script first: `npx ts-node train_kol_classifier.ts`');
    return;
  }

  try {
    // Step 1: Load the trained model
    console.log('📦 Step 1: Loading trained model...');
    const model = await Model.load(modelPath, defaultLogger);
    console.log(`✅ Model loaded successfully from: ${modelPath}`);
    
    // Display model metadata
    const metadata = model.getMetadata();
    console.log('\n📋 Model Metadata:');
    console.log(`- Task Type: ${metadata.taskType}`);
    console.log(`- Classes: ${metadata.classes?.join(', ') || 'N/A'}`);
    console.log(`- Training Accuracy: ${metadata.metrics?.accuracy ? (metadata.metrics.accuracy * 100).toFixed(2) + '%' : 'N/A'}`);
    console.log();

    // Step 2: Test with individual predictions
    console.log('🎯 Step 2: Testing individual predictions...');
    
    // Create sample test cases representing different market conditions
    const testSamples: TestSample[] = [
      {
        // High volatility, bullish momentum - should predict "good"
        candle_count: 10,
        market_cap_candle_count: 10,
        price_at_buy: 0.00005,
        avg_price: 0.000055,
        total_volume_usd: 50000,
        market_cap_at_buy: 50000,
        avg_market_cap: 55000,
        last_available_market_cap: 60000,
        price_change_1h_percent: 5.2,
        price_change_4h_percent: 12.8,
        price_change_24h_percent: 23.4,
        market_cap_change_1h_percent: 5.5,
        market_cap_change_4h_percent: 13.1,
        market_cap_change_24h_percent: 25.5,
        price_volatility_1h: 15.2,
        price_volatility_4h: 18.7,
        price_volatility_24h: 22.3,
        market_cap_volatility_1h: 15.5,
        market_cap_volatility_4h: 19.1,
        market_cap_volatility_24h: 23.2,
        price_atr_24h: 0.000012,
        market_cap_atr_24h: 12000,
        price_range_24h_percent: 45.2,
        market_cap_range_24h_percent: 46.1,
        price_rsi: 65.5,
        market_cap_rsi: 67.2,
        price_macd: 0.000002,
        market_cap_macd: 2500,
        price_sma_20: 0.000048,
        market_cap_sma_20: 48000,
        price_ema_20: 0.000049,
        market_cap_ema_20: 49000,
        price_bollinger_position: 0.8,
        market_cap_bollinger_position: 0.75,
        price_momentum_score: 3.5,
        market_cap_momentum_score: 3.2,
        buy_volume_ratio_24h: 0.7,
        volume_trend_24h: 2.1,
        buy_pressure: 1.2,
        volume_price_correlation: 0.8
      },
      {
        // Low volatility, bearish momentum - should predict "bad"
        candle_count: 5,
        market_cap_candle_count: 5,
        price_at_buy: 0.00003,
        avg_price: 0.000025,
        total_volume_usd: 1000,
        market_cap_at_buy: 30000,
        avg_market_cap: 25000,
        last_available_market_cap: 20000,
        price_change_1h_percent: -2.1,
        price_change_4h_percent: -8.5,
        price_change_24h_percent: -18.2,
        market_cap_change_1h_percent: -2.3,
        market_cap_change_4h_percent: -9.1,
        market_cap_change_24h_percent: -15.5,
        price_volatility_1h: 5.2,
        price_volatility_4h: 7.8,
        price_volatility_24h: 12.1,
        market_cap_volatility_1h: 5.1,
        market_cap_volatility_4h: 8.2,
        market_cap_volatility_24h: 11.8,
        price_atr_24h: 0.000005,
        market_cap_atr_24h: 5000,
        price_range_24h_percent: 25.1,
        market_cap_range_24h_percent: 24.8,
        price_rsi: 25.5,
        market_cap_rsi: 23.2,
        price_macd: -0.000001,
        market_cap_macd: -1200,
        price_sma_20: 0.000035,
        market_cap_sma_20: 35000,
        price_ema_20: 0.000032,
        market_cap_ema_20: 32000,
        price_bollinger_position: 0.2,
        market_cap_bollinger_position: 0.15,
        price_momentum_score: -2.1,
        market_cap_momentum_score: -2.5,
        buy_volume_ratio_24h: 0.3,
        volume_trend_24h: -1.5,
        buy_pressure: -0.8,
        volume_price_correlation: -0.4
      }
    ];

    console.log('Testing sample predictions:');
    console.log('-'.repeat(60));

    for (let i = 0; i < testSamples.length; i++) {
      const sample = testSamples[i];
      if (!sample) continue; // Skip if sample is undefined
      
      // const prediction = model.predict(sample);
      
      console.log(`\nSample ${i + 1}:`);
      console.log(`- Market Cap: $${sample.market_cap_at_buy.toLocaleString()}`);
      console.log(`- 24h Change: ${sample.price_change_24h_percent.toFixed(1)}%`);
      console.log(`- RSI: ${sample.price_rsi.toFixed(1)}`);
      console.log(`- Volume Trend: ${sample.volume_trend_24h.toFixed(1)}`);
      console.log(`- Volatility 24h: ${sample.price_volatility_24h.toFixed(1)}%`);
      // console.log(`- Prediction: ${prediction.predicted} (confidence: ${(prediction.confidence * 100).toFixed(1)}%)`);
      
      // if (prediction.probabilities) {
      //   const probabilities = Object.entries(prediction.probabilities)
      //     .map(([cls, prob]) => `${cls}: ${(prob * 100).toFixed(1)}%`)
      //     .join(', ');
      //   console.log(`- Probabilities: {${probabilities}}`);
      // }
    }

    console.log('\n' + '-'.repeat(60));

    // Step 3: Batch testing on dataset (if available)
    if (fs.existsSync(testDataPath)) {
      console.log('\n📊 Step 3: Batch testing on dataset...');
      
      const dataLoader = new DataLoader(defaultLogger);
      const testData = await dataLoader.loadCSV(testDataPath);

      if (testData && testData.length > 0) {
        // Take a sample for testing (first 100 records or available)
        const sampleSize = Math.min(100, testData.length);
        const testSample = testData.slice(0, sampleSize);
        //console.log("testSample", testSample[3]);
        console.log(`Testing on ${sampleSize} records from dataset...`);
        console.log("testSample", testSample[24]);
        
        const batchResult = model.predictBatch(testSample as RawDataRecord[]);
        const singleResult = model.predictWithProbabilities(testSample[24] as RawDataRecord);
        const transformedResult = model.transform(testSample[24] as RawDataRecord);
        console.log("transformedResult", transformedResult);
        console.log("singleResult", singleResult);
        //console.log("batchResult", batchResult);
        
        console.log(`\n📈 Batch Prediction Results:`);
        console.log(`- Total predictions: ${batchResult.length}`);
        
        // Label mapping for readability
        const labelMap: { [key: string]: string } = {
          '0': 'bad',
          '1': 'good', 
          '2': 'neutral',
          'bad': 'bad',
          'good': 'good',
          'neutral': 'neutral'
        };
        
        // Calculate class distribution
        const classDistribution: { [key: string]: number } = {};
        batchResult.forEach(prediction => {
          const predStr = String(prediction);
          //console.log("predStr", predStr);
          const readableLabel = labelMap[predStr] || predStr;
          classDistribution[readableLabel] = (classDistribution[readableLabel] || 0) + 1;
        });
        
        console.log(`- Class distribution:`);
        Object.entries(classDistribution).forEach(([cls, count]) => {
          const percentage = ((count / batchResult.length) * 100).toFixed(1);
          console.log(`  - ${cls}: ${count} (${percentage}%)`);
        });

        // Show detailed comparison for first few samples
        console.log(`\n📋 Sample Predictions (Expected vs Actual):`);
        console.log('-'.repeat(50));
        
        const detailSamples = Math.min(100, testSample.length);
        for (let i = 0; i < detailSamples; i++) {
          const sample = testSample[i];
          const prediction = batchResult[i];
          
          if (sample && sample.performance_label !== undefined) {
            const expectedLabel = labelMap[String(sample.performance_label)] || String(sample.performance_label);
            const actualLabel = labelMap[String(prediction)] || String(prediction);
            const isCorrect = expectedLabel === actualLabel ? '✅' : '❌';
            
            console.log(`${String(i + 1).padStart(2)}. Expected: ${expectedLabel.padEnd(8)} | Actual: ${actualLabel.padEnd(8)} ${isCorrect}`);
          }
        }

        // Calculate accuracy if we have actual labels
        const actualLabels = testSample
          .map(record => record.performance_label)
          .filter(label => label !== undefined && label !== null);

        if (actualLabels.length > 0) {
          const predictions = batchResult.map(pred => labelMap[String(pred)] || String(pred));
          const expectedLabels = actualLabels.map(label => labelMap[String(label)] || String(label));
          
          const matches = predictions
            .slice(0, expectedLabels.length)
            .filter((pred, idx) => pred === expectedLabels[idx]);
          
          const accuracy = matches.length / expectedLabels.length;
          
          console.log(`\n✅ Test Accuracy: ${(accuracy * 100).toFixed(2)}% (${matches.length}/${expectedLabels.length})`);
          
          // Show confusion matrix summary
          const confusionData: { [key: string]: { [key: string]: number } } = {};
          expectedLabels.forEach((expected, idx) => {
            const actual = predictions[idx];
            if (expected && actual) { // Ensure both are defined
              if (!confusionData[expected]) {
                confusionData[expected] = {};
              }
              const expectedRow = confusionData[expected]!; // Non-null assertion since we just created it
              expectedRow[actual] = (expectedRow[actual] || 0) + 1;
            }
          });
          
          console.log(`\n📊 Confusion Matrix Summary:`);
          Object.entries(confusionData).forEach(([expected, actuals]) => {
            const total = Object.values(actuals).reduce((sum, count) => sum + count, 0);
            console.log(`Expected ${expected}:`);
            Object.entries(actuals).forEach(([actual, count]) => {
              const percentage = ((count / total) * 100).toFixed(1);
              console.log(`  → Predicted ${actual}: ${count} (${percentage}%)`);
            });
          });
        }

      } else {
        console.log('❌ Could not load test data or dataset is empty');
      }
    } else {
      console.log('\n⚠️  Test dataset not found, skipping batch testing');
      console.log(`Expected file: ${testDataPath}`);
    }

    // Step 4: Model insights
    console.log('\n🔍 Step 4: Model Analysis...');
    
    console.log('\n📊 Model Summary:');
    console.log(`- Model successfully loaded and tested`);
    console.log(`- Supports ${metadata.classes?.length || 'unknown'} classes: ${metadata.classes?.join(', ') || 'N/A'}`);
    console.log(`- Ready for production use on OHLCV features`);
    
    console.log('\n💡 Usage Tips:');
    console.log('1. Higher RSI values (>50) typically indicate bullish momentum');
    console.log('2. Positive MACD values suggest upward price trends');
    console.log('3. Bollinger position >0.5 indicates price above moving average');
    console.log('4. Volume correlation >0.5 suggests price-volume alignment');
    console.log('5. Monitor prediction confidence - lower confidence may need human review');

    console.log('\n🎯 Integration Examples:');
    console.log('```typescript');
    console.log('// Load model');
    console.log('const model = await Model.load("./models/kol_ohlcv_classification_model");');
    console.log('');
    console.log('// Predict on new data');
    console.log('const prediction = model.predict({');
    console.log('  token_mint: "...",');
    console.log('  price_momentum_direction: "bullish",');
    console.log('  price_rsi: 65.5,');
    console.log('  // ... other features');
    console.log('});');
    console.log('');
    console.log('console.log(prediction.predicted); // "good" or "bad"');
    console.log('console.log(prediction.confidence); // 0.0 - 1.0');
    console.log('```');

    console.log('\n✅ Testing completed successfully! 🎉');

  } catch (error) {
    console.error('❌ Testing failed:', error);
    
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Ensure the model was trained successfully');
    console.log('2. Check that model files exist in the specified path');
    console.log('3. Verify test data format matches training data');
    console.log('4. Ensure all required features are present in test samples');
    
    throw error;
  }
}

// Handle errors and run the testing
if (require.main === module) {
  testKolClassifier()
    .then(() => {
      console.log('\n🏁 Testing script completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Testing script failed:', error);
      process.exit(1);
    });
}

export { testKolClassifier }; 