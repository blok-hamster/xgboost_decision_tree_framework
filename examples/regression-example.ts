/**
 * @fileoverview Regression Example - XGBoost Decision Tree Toolkit
 * 
 * This example demonstrates how to use the regression capabilities of the
 * XGBoost Decision Tree Toolkit to predict continuous values (house prices)
 * using a comprehensive dataset with 10 features (5 categorical + 5 numeric),
 * including comprehensive tree visualization features.
 */

import {
  createRegressionTrainer,
  Model,
  Metrics,
  validateModelConfig,
  DEFAULT_CONFIG,
} from '../src/index';

/**
 * Main regression example
 */
async function runRegressionExample() {
  console.log('🏠 XGBoost Regression Example - House Price Prediction (10 Features) with Tree Visualization');
  console.log('====================================================================================');
  
  // 1. Generate sample housing data
  console.log('\n📊 Step 1: Generating sample housing data...');
  const housingData = generateHousingData(1000);
  console.log(`Generated ${housingData.length} housing samples`);
  console.log('Sample data:', housingData.slice(0, 3));
  
  // 2. Create regression trainer
  console.log('\n🏗️ Step 2: Creating regression trainer...');
  const trainer = createRegressionTrainer(
    ['location', 'type', 'condition', 'garage', 'heating_type'], // categorical features
    ['area', 'bedrooms', 'bathrooms', 'age', 'yard_size'], // numeric features
    'price' // target variable
  );
  
  // 3. Validate configuration
  console.log('\n✅ Step 3: Validating configuration...');
  const validation = validateModelConfig({
    categoricalFeatures: ['location', 'type', 'condition', 'garage', 'heating_type'],
    numericFeatures: ['area', 'bedrooms', 'bathrooms', 'age', 'yard_size'],
    target: 'price',
    taskType: 'regression',
    xgbParams: DEFAULT_CONFIG.xgboost.regression,
  });
  
  console.log('Configuration validation:', validation);
  
  if (!validation.isValid) {
    console.error('❌ Configuration validation failed:', validation.errors);
    return;
  }
  
  // 4. Load and train model
  console.log('\n🎯 Step 4: Training regression model...');
  const startTime = performance.now();
  
  try {
    await trainer.loadRaw(housingData);
    
    await trainer.train({
      useCrossValidation: true,
      nFolds: 10,
      testRatio: 0.2,
      taskType: 'regression',
      
    });
    
    const endTime = performance.now();
    console.log(`✅ Model trained successfully in ${(endTime - startTime).toFixed(2)}ms`);
    
    // 5. Evaluate model
    console.log('\n📈 Step 5: Evaluating model performance...');
    const metrics = trainer.evaluate();
    console.log('Regression Metrics:');
    console.log(`  • RMSE: ${metrics.rmse?.toFixed(2)}`);
    console.log(`  • MAE: ${metrics.mae?.toFixed(2)}`);
    console.log(`  • R²: ${metrics.r2?.toFixed(4)}`);
    console.log(`  • MAPE: ${metrics.mape?.toFixed(2)}%`);
    
    // 6. Tree Visualization - Get tree statistics
    console.log('\n🌳 Step 6: Analyzing tree structure...');
    const treeStats = trainer.getTreeStatistics();
    console.log(`Model contains ${treeStats.treeCount} decision trees`);
    treeStats.trees.forEach((tree: { index: number; maxDepth: number; nodeCount: number; leafCount: number; }, index: number) => {
      console.log(`  Tree ${index}: ${tree.nodeCount} nodes, max depth ${tree.maxDepth}, ${tree.leafCount} leaves`);
    });
    
    // 7. Text visualization
    console.log('\n📝 Step 7: Text tree visualization...');
    const textViz = trainer.visualizeTree('text', {
      treeIndex: 0,
      includeFeatureNames: true,
      precision: 2,
      includeValues: true,
    });
    console.log('Decision Tree Structure (Full Depth):');
    console.log(textViz.content);
    console.log(`Tree metadata: nodes=${textViz.metadata.nodeCount}, depth=${textViz.metadata.maxDepth}, leaves=${textViz.metadata.leafCount}`);
    
    // 8. Compare different tree depths
    console.log('\n🔍 Step 8: Comparing tree visualizations at different depths...');
    const depths = [2, 3, 4];
    for (const depth of depths) {
      const depthViz = trainer.visualizeTree('text', {
        treeIndex: 0,
        includeFeatureNames: true,
        maxDepth: depth,
        precision: 2,
      });
      console.log(`\n--- Tree depth ${depth} (${depthViz.metadata.nodeCount} nodes shown) ---`);
      console.log(depthViz.content);
    }
    
    // 9. Save model and tree visualizations
    console.log('\n💾 Step 9: Saving trained model and tree visualizations...');
    const modelPath = './models/housing_regression_model';
    await trainer.save(modelPath);
    console.log('✅ Model saved successfully');
    
    // Save tree visualizations to the same directory as the model
    console.log('\n🌳 Saving tree visualizations to model directory...');
    
    // Save HTML visualization
    await trainer.saveTreeVisualization(`${modelPath}/tree_visualization.html`, 'html', {
      includeFeatureNames: true,
      precision: 2,
      includeValues: true,
    });
    console.log('✅ HTML visualization saved to model directory');
    
    // Save SVG visualization
    await trainer.saveTreeVisualization(`${modelPath}/tree_visualization.svg`, 'svg', {
      includeFeatureNames: true,
      precision: 2,
      includeValues: true,
    });
    console.log('✅ SVG visualization saved to model directory');
    
    // Save JSON visualization for custom analysis
    await trainer.saveTreeVisualization(`${modelPath}/tree_visualization.json`, 'json', {
      includeFeatureNames: true,
      precision: 4,
      includeValues: true,
    });
    console.log('✅ JSON visualization saved to model directory');
    
    // Save UML Interactive visualization
    await trainer.saveTreeVisualization(`${modelPath}/tree_uml_interactive.html`, 'uml', {
      includeFeatureNames: true,
      precision: 2,
      includeValues: true,
    });
    console.log('✅ UML Interactive visualization saved to model directory');
    
    // 10. Load model and demonstrate visualization
    console.log('\n🔮 Step 10: Loading model and demonstrating visualization...');
    const model = await Model.load('./models/housing_regression_model');
    console.log('Model loaded:', model.getTaskType());
    console.log('Target stats:', model.getTargetStats());
    
    // Visualize loaded model
    const loadedModelViz = model.visualizeTree('text', {
      treeIndex: 0,
      includeFeatureNames: true,
      precision: 2,
    });
    console.log('\nLoaded model tree visualization (Full Depth):');
    console.log(loadedModelViz.content);
    
    // Get tree statistics from loaded model
    const loadedStats = model.getTreeStatistics();
    console.log(`\nLoaded model has ${loadedStats.treeCount} trees`);
    
    // 12. Make predictions
    console.log('\n🎲 Step 12: Making predictions...');
    const testHouses = [
      {
        location: 'Downtown',
        type: 'Apartment',
        condition: 'Excellent',
        garage: 'Single',
        heating_type: 'Gas',
        area: 1200,
        bedrooms: 2,
        bathrooms: 2,
        age: 5,
        yard_size: 500,
      },
      {
        location: 'Suburb',
        type: 'House',
        condition: 'Good',
        garage: 'Double',
        heating_type: 'Heat Pump',
        area: 2000,
        bedrooms: 4,
        bathrooms: 3,
        age: 10,
        yard_size: 2500,
      },
      {
        location: 'City',
        type: 'Condo',
        condition: 'Fair',
        garage: 'None',
        heating_type: 'Electric',
        area: 800,
        bedrooms: 1,
        bathrooms: 1,
        age: 20,
        yard_size: 200,
      },
    ];
    
    console.log('Predictions:');
    testHouses.forEach((house, i) => {
      const prediction = model.predictValue(house);
      console.log(`  House ${i + 1}: $${prediction.toFixed(2)}`);
      console.log(`    Features: ${house.area}sqft, ${house.bedrooms}BR/${house.bathrooms}BA, ${house.age}y old`);
      console.log(`    Garage: ${house.garage}, Heating: ${house.heating_type}, Yard: ${house.yard_size}sqft`);
    });
    
    // 13. Batch predictions
    console.log('\n🔄 Step 13: Batch predictions...');
    const batchPredictions = model.predictValueBatch(testHouses);
    console.log('Batch predictions:', batchPredictions.map(p => `$${p.toFixed(2)}`));
    
    // 14. Advanced tree visualization analysis
    console.log('\n🎨 Step 14: Advanced tree visualization analysis...');
    
    // Compare visualization formats
    const formats: Array<'text' | 'json' | 'html' | 'svg' | 'uml'> = ['text', 'json', 'html', 'svg', 'uml'];
    
    console.log('Visualization format comparison:');
    for (const format of formats) {
      const viz = trainer.visualizeTree(format, {
        treeIndex: 0,
        includeFeatureNames: true,
        maxDepth: 3,
        precision: 2,
      });
      
      console.log(`  ${format.toUpperCase()}: ${viz.content.length} chars, ${viz.metadata.nodeCount} nodes`);
    }
    
    // Multiple tree analysis (if multiple trees exist)
    if (treeStats.treeCount > 1) {
      console.log('\n🌲 Multiple tree analysis:');
      for (let i = 0; i < Math.min(3, treeStats.treeCount); i++) {
        const treeViz = trainer.visualizeTree('text', {
          treeIndex: i,
          includeFeatureNames: true,
          maxDepth: 2,
          precision: 2,
        });
        console.log(`\n--- Tree ${i} (${treeViz.metadata.nodeCount} nodes) ---`);
        console.log(treeViz.content);
      }
    }
    
    // 15. Performance test
    console.log('\n⚡ Step 15: Performance testing...');
    const testData = generateHousingData(100);
    const perfResults = performanceTest(model, testData);
    console.log('Performance Results:');
    console.log(`  • Samples per second: ${perfResults.samplesPerSecond.toFixed(2)}`);
    console.log(`  • Average latency: ${perfResults.averageLatency.toFixed(2)}ms`);
    console.log(`  • Total time: ${perfResults.totalTime.toFixed(2)}ms`);
    
    // 16. Regression metrics comparison
    console.log('\n📊 Step 16: Regression metrics comparison...');
    const predictions = testData.map(house => model.predictValue(house));
    const actuals = testData.map(house => house.price);
    
    const regressionMetrics = Metrics.calculateRegressionMetrics(predictions, actuals);
    console.log('Detailed Regression Metrics:');
    console.log(`  • MSE: ${regressionMetrics.mse.toFixed(2)}`);
    console.log(`  • RMSE: ${regressionMetrics.rmse.toFixed(2)}`);
    console.log(`  • MAE: ${regressionMetrics.mae.toFixed(2)}`);
    console.log(`  • R²: ${regressionMetrics.r2.toFixed(4)}`);
    console.log(`  • MAPE: ${regressionMetrics.mape.toFixed(2)}%`);
    console.log(`  • Median AE: ${regressionMetrics.medianAe.toFixed(2)}`);
    
    console.log('\n🎉 Regression example with tree visualization completed successfully!');
    console.log('\n📝 Summary:');
    console.log('- Generated synthetic housing data with 10 features (5 categorical + 5 numeric)');
    console.log('- Categorical: location, type, condition, garage, heating_type');
    console.log('- Numeric: area, bedrooms, bathrooms, age, yard_size');
    console.log('- Trained XGBoost regression model');
    console.log('- Analyzed decision tree structure and statistics');
    console.log('- Generated tree visualizations in multiple formats');
    console.log('- Compared different tree depths and visualization formats');
    console.log('- Demonstrated model persistence with visualization support');
    console.log('- Evaluated with comprehensive regression metrics');
    console.log('- Demonstrated single and batch predictions');
    console.log('- Measured inference performance');
    
    console.log('\n📋 Files created:');
    console.log('- models/housing_regression_model/tree_visualization.html - Interactive HTML tree visualization');
    console.log('- models/housing_regression_model/tree_visualization.svg - High-quality SVG tree diagram');
    console.log('- models/housing_regression_model/tree_visualization.json - Structured JSON tree data');
    console.log('- models/housing_regression_model/tree_uml_interactive.html - UML-style interactive tree diagram');
    console.log('- models/housing_regression_model/ - Complete saved model with visualizations');
    
    console.log('\n💡 Tree visualization usage tips:');
    console.log('- Open models/housing_regression_model/tree_visualization.html in a web browser for interactive exploration');
    console.log('- Open models/housing_regression_model/tree_uml_interactive.html for UML-style interactive tree with collapsible nodes');
    console.log('- Use SVG files for high-quality printing or embedding in documents');
    console.log('- JSON format is perfect for custom analysis and visualization tools');
    console.log('- Adjust maxDepth parameter to focus on the most important decision nodes');
    console.log('- Use includeFeatureNames: true for better interpretability');
    console.log('- Tree depth typically correlates with model complexity and potential overfitting');
    console.log('- UML format provides the most interactive experience with expand/collapse controls');
    
  } catch (error) {
    console.error('❌ Error during training:', error);
  }
}

/**
 * Generates synthetic housing data for regression training
 */
function generateHousingData(count: number): any[] {
  const locations = ['Downtown', 'Suburb', 'City', 'Countryside', 'Beach'];
  const types = ['House', 'Apartment', 'Condo', 'Townhouse', 'Villa'];
  const conditions = ['Excellent', 'Good', 'Fair', 'Poor'];
  const garages = ['None', 'Single', 'Double', 'Triple'];
  const heatingTypes = ['Gas', 'Electric', 'Oil', 'Solar', 'Heat Pump'];
  
  const data = [];
  
  for (let i = 0; i < count; i++) {
    const location = locations[Math.floor(Math.random() * locations.length)];
    const type = types[Math.floor(Math.random() * types.length)];
    const condition = conditions[Math.floor(Math.random() * conditions.length)];
    const garage = garages[Math.floor(Math.random() * garages.length)];
    const heating_type = heatingTypes[Math.floor(Math.random() * heatingTypes.length)];
    
    const area = Math.floor(Math.random() * 2000) + 500; // 500-2500 sqft
    const bedrooms = Math.floor(Math.random() * 5) + 1; // 1-5 bedrooms
    const bathrooms = Math.floor(Math.random() * 3) + 1; // 1-3 bathrooms
    const age = Math.floor(Math.random() * 30); // 0-30 years old
    const yard_size = Math.floor(Math.random() * 5000) + 100; // 100-5100 sqft yard
    
    // Calculate price based on features with some realistic logic
    let basePrice = 100000;
    
    // Area factor
    basePrice += area * 150;
    
    // Bedroom factor
    basePrice += bedrooms * 15000;
    
    // Bathroom factor
    basePrice += bathrooms * 10000;
    
    // Age factor (newer = more expensive)
    basePrice -= age * 2000;
    
    // Yard size factor
    basePrice += yard_size * 20;
    
    // Location factor
    const locationMultiplier = {
      'Downtown': 1.4,
      'City': 1.2,
      'Beach': 1.3,
      'Suburb': 1.0,
      'Countryside': 0.8,
    }[location!] || 1.0;
    basePrice *= locationMultiplier;
    
    // Type factor
    const typeMultiplier = {
      'Villa': 1.3,
      'House': 1.1,
      'Townhouse': 1.0,
      'Condo': 0.9,
      'Apartment': 0.8,
    }[type!] || 1.0;
    basePrice *= typeMultiplier;
    
    // Condition factor
    const conditionMultiplier = {
      'Excellent': 1.2,
      'Good': 1.0,
      'Fair': 0.9,
      'Poor': 0.7,
    }[condition!] || 1.0;
    basePrice *= conditionMultiplier;
    
    // Garage factor
    const garageMultiplier = {
      'Triple': 1.15,
      'Double': 1.10,
      'Single': 1.05,
      'None': 1.0,
    }[garage!] || 1.0;
    basePrice *= garageMultiplier;
    
    // Heating type factor
    const heatingMultiplier = {
      'Solar': 1.08,
      'Heat Pump': 1.05,
      'Gas': 1.02,
      'Electric': 1.0,
      'Oil': 0.98,
    }[heating_type!] || 1.0;
    basePrice *= heatingMultiplier;
    
    // Add some random noise
    const noise = (Math.random() - 0.5) * 0.2; // ±10% noise
    const price = Math.round(basePrice * (1 + noise));
    
    data.push({
      location,
      type,
      condition,
      garage,
      heating_type,
      area,
      bedrooms,
      bathrooms,
      age,
      yard_size,
      price,
    });
  }
  
  return data;
}

/**
 * Performance test function for regression models
 */
function performanceTest(model: any, testData: any[]): {
  samplesPerSecond: number;
  averageLatency: number;
  totalTime: number;
  sampleCount: number;
} {
  const startTime = performance.now();
  
  // Test regression performance
  for (const sample of testData) {
    model.predictValue(sample);
  }
  
  const endTime = performance.now();
  const totalTime = endTime - startTime;
  const averageLatency = totalTime / testData.length;
  const samplesPerSecond = (testData.length / totalTime) * 1000;
  
  return {
    samplesPerSecond,
    averageLatency,
    totalTime,
    sampleCount: testData.length,
  };
}

// Run the example
if (require.main === module) {
  runRegressionExample().catch(console.error);
}

export { runRegressionExample, generateHousingData }; 