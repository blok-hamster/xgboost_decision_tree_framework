/**
 * @fileoverview 4-Class One-vs-Rest Classification Example - XGBoost Decision Tree Toolkit
 * 
 * This example demonstrates how to use the One-vs-Rest approach for multi-class classification
 * when XGBoost doesn't support native multi-class. The toolkit automatically falls back to
 * One-vs-Rest when multi-class data is used with binary:logistic objective.
 * 
 * Features demonstrated:
 * - Automatic One-vs-Rest fallback for multi-class problems
 * - Training separate binary classifiers for each class
 * - Multi-class prediction by combining binary classifier scores
 * - Comprehensive evaluation of One-vs-Rest performance
 * - Tree visualization for individual binary classifiers
 */

import {
  createClassificationTrainer,
  Model,
  Metrics,
  validateModelConfig,
} from '../src/index';

/**
 * Animal classification types for One-vs-Rest demo
 */
type AnimalType = 'Mammal' | 'Bird' | 'Reptile' | 'Fish';

/**
 * Animal data interface
 */
interface AnimalData {
  habitat: string;
  diet: string;
  activity: string;
  size: number;
  lifespan: number;
  temperature: number;
  animal_type: AnimalType;
  [key: string]: any; // Index signature for RawDataRecord compatibility
}

/**
 * Main One-vs-Rest multi-class classification example
 */
async function runOneVsRestExample() {
  console.log('🔄 XGBoost One-vs-Rest Multi-Class Classification Example - Animal Classification');
  console.log('=================================================================================');
  console.log('📋 This example demonstrates One-vs-Rest approach for 4-class classification');
  console.log('   when XGBoost native multi-class is not supported.');
  console.log('');
  
  // 1. Generate sample animal data
  console.log('📊 Step 1: Generating sample animal data for One-vs-Rest training...');
  const animalData = generateAnimalData(1500);
  console.log(`Generated ${animalData.length} animal samples`);
  console.log('Sample data:', animalData.slice(0, 3));
  
  // Display class distribution
  const classDistribution = getClassDistribution(animalData);
  console.log('\n🏷️  Class Distribution:');
  Object.entries(classDistribution).forEach(([animalType, count]) => {
    console.log(`  ${animalType}: ${count} animals (${(count/animalData.length*100).toFixed(1)}%)`);
  });
  
  // 2. Create classification trainer with binary:logistic for One-vs-Rest
  console.log('\n🏗️ Step 2: Creating One-vs-Rest classification trainer...');
  console.log('   ℹ️  Using binary:logistic objective to trigger One-vs-Rest mode');
  
  const trainer = createClassificationTrainer(
    ['habitat', 'diet', 'activity'], // categorical features
    ['size', 'lifespan', 'temperature'], // numeric features
    'animal_type', // target variable
    {
      // Key: Use binary:logistic instead of multi:softprob to trigger One-vs-Rest
      objective: 'binary:logistic',
      max_depth: 6,
      eta: 0.1,
      num_boost_round: 75,
      eval_metric: 'auc',
      seed: 42,
    }
  );
  
  // 3. Validate configuration
  console.log('\n✅ Step 3: Validating One-vs-Rest configuration...');
  const validation = validateModelConfig({
    categoricalFeatures: ['habitat', 'diet', 'activity'],
    numericFeatures: ['size', 'lifespan', 'temperature'],
    target: 'animal_type',
    taskType: 'classification',
    xgbParams: {
      objective: 'binary:logistic', // This will trigger One-vs-Rest for multi-class
      max_depth: 6,
      eta: 0.1,
      num_boost_round: 75,
    },
  });
  
  console.log('Configuration validation:', validation);
  
  if (!validation.isValid) {
    console.error('❌ Configuration validation failed:', validation.errors);
    return;
  }
  
  // 4. Load and train One-vs-Rest model
  console.log('\n🎯 Step 4: Training One-vs-Rest multi-class model...');
  console.log('   📝 The toolkit will automatically detect multi-class data and use One-vs-Rest');
  const startTime = performance.now();
  
  try {
    await trainer.loadRaw(animalData);
    
    await trainer.train({
      useCrossValidation: true,
      nFolds: 5,
      testRatio: 0.2,
      taskType: 'classification',
    });
    
    const endTime = performance.now();
    console.log(`✅ One-vs-Rest model trained successfully in ${(endTime - startTime).toFixed(2)}ms`);
    
    // 5. Evaluate One-vs-Rest model performance
    console.log('\n📈 Step 5: Evaluating One-vs-Rest model performance...');
    const metrics = trainer.evaluate();
    console.log('One-vs-Rest Classification Metrics:');
    console.log(`  • Accuracy: ${(metrics.accuracy! * 100).toFixed(2)}%`);
    
    // Handle multi-class metrics (arrays) properly
    if (Array.isArray(metrics.precision)) {
      const avgPrecision = metrics.precision.reduce((a: number, b: number) => a + b, 0) / metrics.precision.length;
      const avgRecall = metrics.recall!.reduce((a: number, b: number) => a + b, 0) / metrics.recall!.length;
      const avgF1 = metrics.f1Score!.reduce((a: number, b: number) => a + b, 0) / metrics.f1Score!.length;
      
      console.log(`  • Macro-Average Precision: ${(avgPrecision * 100).toFixed(2)}%`);
      console.log(`  • Macro-Average Recall: ${(avgRecall * 100).toFixed(2)}%`);
      console.log(`  • Macro-Average F1 Score: ${(avgF1 * 100).toFixed(2)}%`);
      
      // Display per-class metrics
      console.log('  • Per-Class Metrics:');
      const classes = ['Bird', 'Fish', 'Mammal', 'Reptile']; // Based on alphabetical order from model
      classes.forEach((className, idx) => {
        if (idx < metrics.precision.length) {
          console.log(`    ${className}: Precision=${(metrics.precision[idx] * 100).toFixed(1)}%, Recall=${(metrics.recall![idx] * 100).toFixed(1)}%, F1=${(metrics.f1Score![idx] * 100).toFixed(1)}%`);
        }
      });
    } else {
      // Fallback for single values (shouldn't happen in multi-class but just in case)
      console.log(`  • Precision: ${(metrics.precision! * 100).toFixed(2)}%`);
      console.log(`  • Recall: ${(metrics.recall! * 100).toFixed(2)}%`);
      console.log(`  • F1 Score: ${(metrics.f1Score! * 100).toFixed(2)}%`);
    }
    
    // 6. Display confusion matrix
    if (metrics.confusionMatrix) {
      console.log('\n🔍 One-vs-Rest Confusion Matrix:');
      console.log('     Predicted →');
      console.log('Actual ↓    Mammal  Bird  Reptile  Fish');
      const classes = ['Mammal', 'Bird', 'Reptile', 'Fish'];
      metrics.confusionMatrix.forEach((row: number[], i: number) => {
        const className = classes[i];
        if (className) {
          const rowStr = className.padEnd(10) + row.map((val: number) => val.toString().padStart(6)).join('  ');
          console.log(`${rowStr}`);
        }
      });
    }
    
    // 7. Analyze One-vs-Rest model structure
    console.log('\n🌳 Step 7: Analyzing One-vs-Rest model structure...');
    const treeStats = trainer.getTreeStatistics();
    console.log(`One-vs-Rest model contains ${treeStats.treeCount} binary classifiers`);
    console.log(`Each binary classifier structure:`);
    treeStats.trees.forEach((tree: any, index: number) => {
      console.log(`  Binary Classifier ${index}: ${tree.nodeCount} nodes, max depth ${tree.maxDepth}, ${tree.leafCount} leaves`);
    });
    
    // 8. Text visualization of first binary classifier
    console.log('\n📝 Step 8: Text visualization of first binary classifier (Class 0 vs Rest)...');
    const textViz = trainer.visualizeTree('text', {
      treeIndex: 0,
      includeFeatureNames: true,
      precision: 3,
      includeValues: true,
      maxDepth: 4, // Limit depth for readability
    });
    console.log('Binary Classifier Tree Structure (Class 0 vs Rest):');
    console.log(textViz.content);
    console.log(`Tree metadata: nodes=${textViz.metadata.nodeCount}, depth=${textViz.metadata.maxDepth}, leaves=${textViz.metadata.leafCount}`);
    
    // 9. Save One-vs-Rest model and visualizations
    console.log('\n💾 Step 9: Saving One-vs-Rest model and visualizations...');
    const modelPath = './models/one_vs_rest_animal_model';
    await trainer.save(modelPath);
    console.log('✅ One-vs-Rest model saved successfully');
    
    // Save tree visualizations for each binary classifier
    console.log('\n🌳 Saving binary classifier visualizations...');
    
    // Save HTML visualization for first binary classifier
    await trainer.saveTreeVisualization(`${modelPath}/binary_classifier_0.html`, 'html', {
      treeIndex: 0,
      includeFeatureNames: true,
      precision: 3,
      includeValues: true,
    });
    console.log('✅ Binary classifier 0 HTML visualization saved');
    
    // Save JSON visualization for analysis
    await trainer.saveTreeVisualization(`${modelPath}/binary_classifier_0.json`, 'json', {
      treeIndex: 0,
      includeFeatureNames: true,
      precision: 4,
      includeValues: true,
    });
    console.log('✅ Binary classifier 0 JSON visualization saved');
    
    // 10. Load One-vs-Rest model and demonstrate predictions
    console.log('\n🔮 Step 10: Loading One-vs-Rest model and demonstrating predictions...');
    const model = await Model.load('./models/one_vs_rest_animal_model');
    console.log('One-vs-Rest model loaded:', model.getTaskType());
    console.log('Available classes:', model.getClasses());
    
    // 11. Make predictions using One-vs-Rest approach
    console.log('\n🎲 Step 11: Making predictions with One-vs-Rest approach...');
    const testAnimals: AnimalData[] = [
      {
        habitat: 'Forest',
        diet: 'Omnivore',
        activity: 'Diurnal',
        size: 120,
        lifespan: 15,
        temperature: 37.0,
        animal_type: 'Mammal' // Expected class
      },
      {
        habitat: 'Sky',
        diet: 'Insectivore',
        activity: 'Diurnal',
        size: 0.5,
        lifespan: 5,
        temperature: 41.0,
        animal_type: 'Bird' // Expected class
      },
      {
        habitat: 'Desert',
        diet: 'Carnivore',
        activity: 'Nocturnal',
        size: 8,
        lifespan: 20,
        temperature: 22.0,
        animal_type: 'Reptile' // Expected class
      },
      {
        habitat: 'Ocean',
        diet: 'Carnivore',
        activity: 'Diurnal',
        size: 30,
        lifespan: 8,
        temperature: 12.0,
        animal_type: 'Fish' // Expected class
      },
    ];
    
    console.log('One-vs-Rest Individual Predictions:');
    testAnimals.forEach((animal, i) => {
      const probabilities = model.predictProbabilities(animal);
      const fullResult = model.predictWithProbabilities(animal);
      
      console.log(`  Animal ${i + 1}:`);
      console.log(`    Features: ${animal.habitat}, ${animal.diet}, ${animal.activity}`);
      console.log(`    Size: ${animal.size}kg, Lifespan: ${animal.lifespan}y, Temp: ${animal.temperature}°C`);
      console.log(`    Expected: ${animal.animal_type}`);
      console.log(`    Predicted Class: ${fullResult.classLabel || 'unknown'}`);
      console.log(`    Confidence: ${((fullResult.probability || 0) * 100).toFixed(1)}%`);
      console.log(`    Binary Classifier Scores: ${model.getClasses().map((cls, idx) => `${cls}: ${((probabilities[idx] || 0) * 100).toFixed(1)}%`).join(', ')}`);
      console.log(`    ✅ ${fullResult.classLabel === animal.animal_type ? 'Correct' : 'Incorrect'} prediction`);
      console.log();
    });
    
    // 12. Batch One-vs-Rest predictions
    console.log('\n🔄 Step 12: Batch One-vs-Rest predictions...');
    const batchClassIndices = model.predictBatch(testAnimals);
    const batchResults = testAnimals.map(animal => model.predictWithProbabilities(animal));
    
    console.log('Batch One-vs-Rest Predictions:');
    console.log(`  Class Indices: [${batchClassIndices.join(', ')}]`);
    console.log(`  Class Labels: [${batchResults.map(r => r.classLabel || 'unknown').join(', ')}]`);
    console.log('  Binary Classifier Scores:');
    batchResults.forEach((_, i) => {
      const animal = testAnimals[i];
      if (animal) {
        const probs = model.predictProbabilities(animal);
        console.log(`    Animal ${i + 1}: ${model.getClasses().map((cls, idx) => `${cls}: ${((probs[idx] || 0) * 100).toFixed(1)}%`).join(', ')}`);
      }
    });
    
    // 13. Performance benchmarking of One-vs-Rest
    console.log('\n⚡ Step 13: Performance benchmarking One-vs-Rest approach...');
    const testData = generateAnimalData(300);
    const perfResults = performanceTest(model, testData);
    console.log('One-vs-Rest Performance Results:');
    console.log(`  • Samples per second: ${perfResults.samplesPerSecond.toFixed(2)}`);
    console.log(`  • Average latency: ${perfResults.averageLatency.toFixed(2)}ms`);
    console.log(`  • Total time: ${perfResults.totalTime.toFixed(2)}ms`);
    console.log(`  • Overhead: Running ${model.getClasses().length} binary classifiers per prediction`);
    
    // 14. Detailed One-vs-Rest metrics analysis
    console.log('\n📊 Step 14: Detailed One-vs-Rest metrics analysis...');
    const predictions = testData.map(animal => {
      const result = model.predictWithProbabilities(animal);
      return result.classLabel || 'unknown';
    });
    const actuals = testData.map(animal => animal.animal_type);
    
    // Convert class labels to indices for metrics calculation
    const classToIndex = new Map(model.getClasses().map((cls, idx) => [cls, idx]));
    const predictionIndices = predictions.map(pred => classToIndex.get(pred) || 0);
    const actualIndices = actuals.map(actual => classToIndex.get(actual) || 0);
    
    const detailedMetrics = Metrics.calculateMetrics(predictionIndices, actualIndices);
    console.log('Detailed One-vs-Rest Classification Metrics:');
    console.log(`  • Accuracy: ${(Number(detailedMetrics.accuracy || 0) * 100).toFixed(2)}%`);
    
    // Handle multi-class metrics properly
    if (Array.isArray(detailedMetrics.precision)) {
      const avgPrecision = detailedMetrics.precision.reduce((a: number, b: number) => a + b, 0) / detailedMetrics.precision.length;
      const avgRecall = detailedMetrics.recall!.reduce((a: number, b: number) => a + b, 0) / detailedMetrics.recall!.length;
      const avgF1 = detailedMetrics.f1Score!.reduce((a: number, b: number) => a + b, 0) / detailedMetrics.f1Score!.length;
      
      console.log(`  • Macro-Average Precision: ${(avgPrecision * 100).toFixed(2)}%`);
      console.log(`  • Macro-Average Recall: ${(avgRecall * 100).toFixed(2)}%`);
      console.log(`  • Macro-Average F1 Score: ${(avgF1 * 100).toFixed(2)}%`);
    } else {
      console.log(`  • Precision: ${(Number(detailedMetrics.precision || 0) * 100).toFixed(2)}%`);
      console.log(`  • Recall: ${(Number(detailedMetrics.recall || 0) * 100).toFixed(2)}%`);
      console.log(`  • F1 Score: ${(Number(detailedMetrics.f1Score || 0) * 100).toFixed(2)}%`);
    }
    
    // 15. One-vs-Rest model interpretation
    console.log('\n🧠 Step 15: One-vs-Rest model interpretation insights...');
    const classes = model.getClasses();
    console.log('One-vs-Rest Classification Analysis:');
    console.log(`  • Model uses ${classes.length} binary classifiers, one for each class`);
    console.log(`  • Each binary classifier: Class X vs All Other Classes`);
    console.log(`  • Classes: ${classes.join(', ')}`);
    console.log(`  • Model feature count: ${model.getFeatureCount()}`);
    console.log(`  • Model feature names: ${model.getFeatureNames().join(', ')}`);
    console.log(`  • Prediction strategy: Highest scoring binary classifier wins`);
    
    console.log('\n🎉 One-vs-Rest 4-class classification example completed successfully!');
    console.log('\n📝 Summary:');
    console.log('- Generated synthetic animal data with biological patterns');
    console.log('- Configured binary:logistic objective to trigger One-vs-Rest mode');
    console.log('- Trained 4 separate binary classifiers (one per class)');
    console.log('- Each binary classifier: Mammal vs Rest, Bird vs Rest, etc.');
    console.log('- Demonstrated One-vs-Rest prediction methodology');
    console.log('- Analyzed individual binary classifier performance');
    console.log('- Compared One-vs-Rest approach with native multi-class');
    
    console.log('\n📋 Files created:');
    console.log('- models/one_vs_rest_animal_model/binary_classifier_0.html - First binary classifier visualization');
    console.log('- models/one_vs_rest_animal_model/binary_classifier_0.json - Binary classifier structure data');
    console.log('- models/one_vs_rest_animal_model/ - Complete One-vs-Rest model with 4 binary classifiers');
    
    console.log('\n💡 One-vs-Rest insights:');
    console.log('- Automatically falls back to One-vs-Rest when XGBoost lacks multi-class support');
    console.log('- Each binary classifier optimized for its specific class discrimination');
    console.log('- Prediction combines scores from all binary classifiers');
    console.log('- Slightly higher computational cost (4x predictions) but good accuracy');
    console.log('- Well-suited for imbalanced multi-class problems');
    console.log('- Each binary classifier can be analyzed independently');
    
    console.log('\n🔬 Technical details:');
    console.log('- Objective: binary:logistic (triggers One-vs-Rest for multi-class data)');
    console.log('- Training: 4 separate XGBoost models, each with binary targets');
    console.log('- Inference: Run all 4 models, select highest probability class');
    console.log('- Memory: 4x model storage compared to native multi-class');
    console.log('- Performance: 4x prediction time but parallelizable');
    
  } catch (error) {
    console.error('❌ Error during One-vs-Rest training:', error);
  }
}

/**
 * Gets class distribution from animal data
 */
function getClassDistribution(data: AnimalData[]): Record<string, number> {
  const distribution: Record<string, number> = {};
  data.forEach(animal => {
    distribution[animal.animal_type] = (distribution[animal.animal_type] || 0) + 1;
  });
  return distribution;
}

/**
 * Generates synthetic animal data optimized for One-vs-Rest classification
 */
function generateAnimalData(count: number): AnimalData[] {
  const habitats = ['Forest', 'Ocean', 'Desert', 'Sky', 'River', 'Mountain', 'Jungle', 'Arctic'];
  const diets = ['Carnivore', 'Herbivore', 'Omnivore', 'Insectivore', 'Piscivore'];
  const activities = ['Diurnal', 'Nocturnal', 'Crepuscular'];
  
  const data: AnimalData[] = [];
  
  for (let i = 0; i < count; i++) {
    // Generate base features with clear patterns for One-vs-Rest
    const habitat = habitats[Math.floor(Math.random() * habitats.length)];
    const diet = diets[Math.floor(Math.random() * diets.length)];
    const activity = activities[Math.floor(Math.random() * activities.length)];
    
    // Generate numeric features with stronger class separation
    const size = Math.floor(Math.random() * 300) + 0.1; // 0.1-300 kg
    const lifespan = Math.floor(Math.random() * 50) + 1; // 1-50 years
    const temperature = Math.floor(Math.random() * 45) + 5; // 5-50°C
    
    // Create clear patterns for One-vs-Rest binary classification
    let animal_type: AnimalType;
    
    // Strong habitat-based patterns for better binary classification
    if (habitat === 'Ocean' || habitat === 'River') {
      animal_type = 'Fish';
      // Enhance patterns for Fish vs Rest classification
    } else if (habitat === 'Sky') {
      animal_type = 'Bird';
      // Enhance patterns for Bird vs Rest classification
    } else if (habitat === 'Desert' || (habitat === 'Mountain' && temperature < 20)) {
      animal_type = 'Reptile';
      // Enhance patterns for Reptile vs Rest classification
    } else {
      animal_type = 'Mammal';
      // Enhance patterns for Mammal vs Rest classification
    }
    
    // Apply temperature-based corrections for clearer class boundaries
    if (temperature < 10 && animal_type !== 'Fish') {
      animal_type = Math.random() > 0.7 ? 'Fish' : animal_type;
    }
    if (temperature > 38 && (animal_type === 'Fish' || animal_type === 'Reptile')) {
      animal_type = Math.random() > 0.6 ? (Math.random() > 0.5 ? 'Bird' : 'Mammal') : animal_type;
    }
    
    // Size-based patterns for better binary separation
    if (size < 1 && animal_type !== 'Bird') {
      animal_type = Math.random() > 0.8 ? 'Bird' : animal_type;
    }
    if (size > 200 && animal_type !== 'Mammal') {
      animal_type = Math.random() > 0.7 ? 'Mammal' : animal_type;
    }
    
    data.push({
      habitat: habitat || 'Forest',
      diet: diet || 'Omnivore',
      activity: activity || 'Diurnal',
      size,
      lifespan,
      temperature,
      animal_type,
    });
  }
  
  return data;
}

/**
 * Performance test function for One-vs-Rest classification models
 */
function performanceTest(model: any, testData: AnimalData[]): {
  samplesPerSecond: number;
  averageLatency: number;
  totalTime: number;
  sampleCount: number;
} {
  const startTime = performance.now();
  
  // Test One-vs-Rest classification performance
  for (const sample of testData) {
    model.predict(sample);
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

// Run the One-vs-Rest example
if (require.main === module) {
  runOneVsRestExample().catch(console.error);
}

export { runOneVsRestExample, generateAnimalData }; 