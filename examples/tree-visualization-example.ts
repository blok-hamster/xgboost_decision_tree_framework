/**
 * @fileoverview Tree Visualization Example - XGBoost Decision Tree Toolkit
 * 
 * This example demonstrates how to visualize decision trees from trained XGBoost models
 * in multiple formats: text, JSON, HTML, and SVG.
 */

import {
  createClassificationTrainer,
  createRegressionTrainer,
  Model,
  createTestDataset,
} from '../src/index';

/**
 * Main tree visualization example
 */
async function runTreeVisualizationExample() {
  console.log('🌳 XGBoost Tree Visualization Example');
  console.log('====================================');
  
  // 1. Train a classification model
  console.log('\n📊 Step 1: Training classification model...');
  const classificationData = createTestDataset(1000, 'binary');
  
  const classificationTrainer = createClassificationTrainer(
    ['category', 'brand', 'color'], // categorical features
    ['price', 'rating', 'sales'], // numeric features
    'target' // target variable
  );
  
  await classificationTrainer.loadRaw(classificationData);
  await classificationTrainer.train({
    useCrossValidation: false,
    testRatio: 0.2,
  });
  
  console.log('✅ Classification model trained');
  
  // 2. Get tree statistics
  console.log('\n📈 Step 2: Getting tree statistics...');
  const stats = classificationTrainer.getTreeStatistics();
  console.log(`Model has ${stats.treeCount} trees`);
  stats.trees.forEach((tree: any, index: number) => {
    console.log(`  Tree ${index}: ${tree.nodeCount} nodes, max depth ${tree.maxDepth}, ${tree.leafCount} leaves`);
  });
  
  // 3. Text visualization
  console.log('\n📝 Step 3: Text visualization...');
  const textViz = classificationTrainer.visualizeTree('text', {
    treeIndex: 0,
    includeFeatureNames: true,
    maxDepth: 3,
    precision: 2,
  });
  console.log('Text visualization:');
  console.log(textViz.content);
  console.log(`\nTree metadata: ${JSON.stringify(textViz.metadata, null, 2)}`);
  
  // 4. Save visualizations to files
  console.log('\n💾 Step 4: Saving visualizations to files...');
  
  // Save HTML visualization
  await classificationTrainer.saveTreeVisualization('./tree_classification.html', 'html', {
    includeFeatureNames: true,
    maxDepth: 4,
    precision: 3,
  });
  console.log('✅ HTML visualization saved to tree_classification.html');
  
  // Save SVG visualization
  await classificationTrainer.saveTreeVisualization('./tree_classification.svg', 'svg', {
    includeFeatureNames: true,
    maxDepth: 3,
    precision: 2,
  });
  console.log('✅ SVG visualization saved to tree_classification.svg');
  
  // Save JSON visualization
  await classificationTrainer.saveTreeVisualization('./tree_classification.json', 'json', {
    includeFeatureNames: true,
    includeValues: true,
    precision: 4,
  });
  console.log('✅ JSON visualization saved to tree_classification.json');
  
  // 5. Train a regression model
  console.log('\n🏠 Step 5: Training regression model...');
  const regressionData = generateRegressionData(1000);
  
  const regressionTrainer = createRegressionTrainer(
    ['location', 'type'], // categorical features
    ['area', 'age'], // numeric features
    'price' // target variable
  );
  
  await regressionTrainer.loadRaw(regressionData);
  await regressionTrainer.train({
    useCrossValidation: false,
    testRatio: 0.2,
  });
  
  console.log('✅ Regression model trained');
  
  // 6. Visualize regression tree
  console.log('\n🌲 Step 6: Visualizing regression tree...');
  const regressionTextViz = regressionTrainer.visualizeTree('text', {
    treeIndex: 0,
    includeFeatureNames: true,
    maxDepth: 3,
    precision: 2,
  });
  console.log('Regression tree visualization:');
  console.log(regressionTextViz.content);
  
  // Save regression tree HTML
  await regressionTrainer.saveTreeVisualization('./tree_regression.html', 'html', {
    includeFeatureNames: true,
    maxDepth: 4,
    precision: 2,
  });
  console.log('✅ Regression HTML visualization saved to tree_regression.html');
  
  // 7. Save and load model, then visualize
  console.log('\n🔄 Step 7: Testing model persistence with visualization...');
  await classificationTrainer.save('./models/classification_viz_model');
  
  const loadedModel = await Model.load('./models/classification_viz_model');
  
  // Visualize loaded model
  const loadedModelViz = loadedModel.visualizeTree('text', {
    treeIndex: 0,
    includeFeatureNames: true,
    maxDepth: 2,
  });
  console.log('Loaded model visualization:');
  console.log(loadedModelViz.content);
  
  // 8. Advanced TreeVisualizer usage
  console.log('\n🔧 Step 8: Advanced TreeVisualizer usage...');
  
  // Get internal model and visualize
  const modelJSON = classificationTrainer.getModelMetadata();
  console.log('Model metadata:');
  console.log(`- Version: ${modelJSON.version}`);
  console.log(`- Task type: ${modelJSON.taskType}`);
  console.log(`- Classes: ${modelJSON.classes?.join(', ')}`);
  console.log(`- Features: ${modelJSON.categoricalFeatures.join(', ')}, ${modelJSON.numericFeatures.join(', ')}`);
  
  // 9. Compare different visualization formats
  console.log('\n🎨 Step 9: Comparing visualization formats...');
  
  const formats: Array<'text' | 'json' | 'html' | 'svg'> = ['text', 'json', 'html', 'svg'];
  
  for (const format of formats) {
    const viz = classificationTrainer.visualizeTree(format, {
      treeIndex: 0,
      includeFeatureNames: true,
      maxDepth: 2,
      precision: 2,
    });
    
    console.log(`${format.toUpperCase()} format: ${viz.content.length} characters`);
    console.log(`  - Tree index: ${viz.metadata.treeIndex}`);
    console.log(`  - Max depth: ${viz.metadata.maxDepth}`);
    console.log(`  - Node count: ${viz.metadata.nodeCount}`);
    console.log(`  - Leaf count: ${viz.metadata.leafCount}`);
  }
  
  console.log('\n🎉 Tree visualization example completed successfully!');
  console.log('\n📋 Files created:');
  console.log('- tree_classification.html - Interactive HTML visualization');
  console.log('- tree_classification.svg - Scalable vector graphics');
  console.log('- tree_classification.json - Structured JSON format');
  console.log('- tree_regression.html - Regression tree visualization');
  console.log('- models/classification_viz_model/ - Saved model with visualization support');
  
  console.log('\n💡 Usage tips:');
  console.log('- Open HTML files in a web browser for interactive visualization');
  console.log('- Use SVG files for high-quality printing or embedding');
  console.log('- JSON format is great for custom visualization tools');
  console.log('- Text format is perfect for quick debugging in terminal');
  console.log('- Adjust maxDepth to focus on important parts of large trees');
  console.log('- Use includeFeatureNames: true for better readability');
}

/**
 * Generates sample regression data for demonstration
 */
function generateRegressionData(count: number): any[] {
  const locations = ['Downtown', 'Suburb', 'City'];
  const types = ['House', 'Apartment', 'Condo'];
  
  const data = [];
  
  for (let i = 0; i < count; i++) {
    const location = locations[Math.floor(Math.random() * locations.length)];
    const type = types[Math.floor(Math.random() * types.length)];
    const area = Math.floor(Math.random() * 1500) + 500; // 500-2000 sqft
    const age = Math.floor(Math.random() * 30); // 0-30 years
    
    // Calculate price based on features
    let price = 100000;
    price += area * 150;
    price -= age * 2000;
    
    if (location === 'Downtown') price *= 1.4;
    if (location === 'City') price *= 1.2;
    
    if (type === 'House') price *= 1.1;
    if (type === 'Apartment') price *= 0.9;
    
    // Add some noise
    price += (Math.random() - 0.5) * 50000;
    
    data.push({
      location,
      type,
      area,
      age,
      price: Math.round(price),
    });
  }
  
  return data;
}

// Run the example
if (require.main === module) {
  runTreeVisualizationExample().catch(console.error);
}

export { runTreeVisualizationExample }; 