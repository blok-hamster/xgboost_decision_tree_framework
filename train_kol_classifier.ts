/**
 * @fileoverview Training script for Solana KOL OHLCV Features Classification
 * 
 * This script trains a classification model to predict trading performance
 * based on OHLCV features extracted from Solana KOL trading data.
 * 
 * Features:
 * - Categorical: token_mint, quote_mint, price_momentum_direction, market_cap_momentum_direction
 * - Numeric: All price/market cap indicators, volatilities, technical indicators
 * - Target: performance_label (good/bad trading outcomes)
 */

import {
  Trainer,
  //FeatureAnalyzer,
  Metrics,
  defaultLogger,
} from './src';
import * as path from 'path';
import * as fs from 'fs';

async function trainKolClassifier(): Promise<void> {
  console.log('🚀 Solana KOL OHLCV Features Classification Training\n');

  // Define feature sets based on the CSV structure
  const categoricalFeatures: string[] = []; // Remove categorical features

  const numericFeatures = [
    // Timestamp and counts
    'candle_count',
    'market_cap_candle_count',
    
    // Price and market cap base metrics
    'price_at_buy',
    'avg_price',
    'total_volume_usd',
    'market_cap_at_buy',
    'avg_market_cap',
    'last_available_market_cap',
    
    // Change percentages
    'market_cap_change_24h_percent',
    'price_change_1h_percent',
    'price_change_4h_percent',
    'price_change_24h_percent',
    'market_cap_change_1h_percent',
    'market_cap_change_4h_percent',
    'market_cap_change_24h_percent',
    
    // Volatility metrics
    'price_volatility_1h',
    'price_volatility_4h',
    'price_volatility_24h',
    'market_cap_volatility_1h',
    'market_cap_volatility_4h',
    'market_cap_volatility_24h',
    
    // ATR and ranges
    'price_atr_24h',
    'market_cap_atr_24h',
    'price_range_24h_percent',
    'market_cap_range_24h_percent',
    
    // Technical indicators
    'price_rsi',
    'market_cap_rsi',
    'price_macd',
    'market_cap_macd',
    'price_sma_20',
    'market_cap_sma_20',
    'price_ema_20',
    'market_cap_ema_20',
    
    // Bollinger bands and momentum
    'price_bollinger_position',
    'market_cap_bollinger_position',
    'price_momentum_score',
    'market_cap_momentum_score',
    
    // Volume metrics
    'buy_volume_ratio_24h',
    'volume_trend_24h',
    'buy_pressure',
    'volume_price_correlation'
  ];

  const target = 'performance_label';
  
  // Define data path - adjust based on your file location
  const dataPath = './data-extraction/output/batch-ohlcv/Cupsey/Cupsey_ohlcv_features_2025-07-18T17-18-07-881Z.csv';
  
  if (!fs.existsSync(dataPath)) {
    console.error(`❌ Data file not found: ${dataPath}`);
    console.log('Please ensure the OHLCV features CSV file is available at the specified path.');
    return;
  }

  console.log('📊 Step 1: Setting up trainer configuration...');
  
  // Create trainer with optimized parameters for financial data classification
  const trainer = new Trainer({
    categoricalFeatures,
    numericFeatures,
    target,
    taskType: 'classification',
    xgbParams: {
      objective: 'binary:logistic', // Assuming binary classification (good/bad)
      max_depth: 8,
      eta: 0.05, // Lower learning rate for financial data
      min_child_weight: 3,
      subsample: 0.8,
      colsample_bytree: 0.8,
      gamma: 1,
      lambda: 1,
      alpha: 0,
      num_boost_round: 400,
      //early_stopping_rounds: 20,
    },
  }, defaultLogger);

  console.log('Configuration:');
  console.log(`- Categorical features: ${categoricalFeatures.length}`);
  console.log(`- Numeric features: ${numericFeatures.length}`);
  console.log(`- Target: ${target}`);
  console.log(`- Data file: ${dataPath}`);
  console.log();

  try {
    // Step 2: Load and analyze data
    console.log('📈 Step 2: Loading OHLCV features data...');
    await trainer.loadCSV(dataPath, {
      parseNumbers: true,
      skipEmptyLines: true,
      headers: true,
      skipRows: 100,
    });

    // Step 3: Train model with cross-validation
    console.log('🎯 Step 4: Training classification model...');
    console.log('This may take several minutes for large datasets...\n');
    
    await trainer.train({
      useCrossValidation: true,
      nFolds: 5,
      testRatio: 0.2,
      shuffle: true,
      seed: 42,
    });

    // Step 4: Evaluate performance
    console.log('📊 Step 5: Evaluating model performance...');
    const metrics = trainer.evaluate();

    // save visualization
    //const visualization = trainer.visualizeTree('uml');
    //console.log("visualization", visualization);
    
    console.log('\n🎉 Training Results:');
    console.log('=' .repeat(50));
    
    if (metrics.accuracy !== undefined) {
      console.log(`Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
    }
    
    if (metrics.precision && metrics.precision.length > 0) {
      console.log(`Precision: [${metrics.precision.map(p => (p * 100).toFixed(1)).join(', ')}]%`);
    }
    
    if (metrics.recall && metrics.recall.length > 0) {
      console.log(`Recall: [${metrics.recall.map(r => (r * 100).toFixed(1)).join(', ')}]%`);
    }
    
    if (metrics.f1Score && metrics.f1Score.length > 0) {
      console.log(`F1-Score: [${metrics.f1Score.map(f => (f * 100).toFixed(1)).join(', ')}]%`);
    }
    
    if (metrics.crossValidationScores) {
      const cvStats = Metrics.crossValidationStats(metrics.crossValidationScores);
      console.log(`Cross-Validation: ${(cvStats.mean * 100).toFixed(2)}% ± ${(cvStats.std * 100).toFixed(2)}%`);
      console.log(`CV Range: ${(cvStats.min * 100).toFixed(2)}% - ${(cvStats.max * 100).toFixed(2)}%`);
    }
    
    console.log('=' .repeat(50));

    // Step 5: Save the trained model
    console.log('\n💾 Step 6: Saving trained model...');
    const modelPath = './models/cupsey_ohlcv_classification_model_v2';
    
    // Ensure models directory exists
    const modelsDir = path.dirname(modelPath);
    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
    }
    
    await trainer.save(modelPath);
    console.log(`✅ Model saved successfully to: ${modelPath}`);
    console.log();

    // Step 6: Display model info
    console.log('📋 Model Information:');
    console.log(`- Model type: XGBoost Classification`);
    console.log(`- Features: ${categoricalFeatures.length + numericFeatures.length} total`);
    console.log(`- Training objective: ${trainer['xgbParams'].objective}`);
    console.log(`- Boosting rounds: ${trainer['xgbParams'].num_boost_round}`);
    console.log(`- Max depth: ${trainer['xgbParams'].max_depth}`);
    console.log(`- Learning rate: ${trainer['xgbParams'].eta}`);
    console.log();

    console.log('🎯 Next Steps:');
    console.log('1. Run the test script: `npx ts-node test_kol_classifier.ts`');
    console.log('2. Use the model for predictions on new OHLCV data');
    console.log('3. Monitor model performance and retrain as needed');
    console.log();

    console.log('✅ Training completed successfully! 🎉');

  } catch (error) {
    console.error('❌ Training failed:', error);
    
    if (error instanceof Error) {
      console.error('Error details:', error.message);
      console.error('Stack trace:', error.stack);
    }
    
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Check that the CSV file exists and has the correct format');
    console.log('2. Ensure all required columns are present in the data');
    console.log('3. Verify that the target column contains valid labels');
    console.log('4. Check for sufficient memory if working with large datasets');
    
    throw error;
  }
}

// Handle errors and run the training
if (require.main === module) {
  trainKolClassifier()
    .then(() => {
      console.log('\n🏁 Script completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Script failed:', error);
      process.exit(1);
    });
}

export { trainKolClassifier }; 