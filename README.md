# Decision Tree Model Toolkit

A comprehensive TypeScript toolkit for training, persisting, loading, and inferring high-performance decision-tree models using XGBoost. Features robust feature encoding, multi-class classification, model consistency, and smooth integration in production pipelines.

## 🚀 Features

- **🔧 Feature Engineering**: Advanced hash-based categorical feature encoding with collision handling
- **📊 Multi-class Classification**: Native XGBoost multi-class support with One-vs-Rest fallback
- **🔒 Deterministic & Reproducible**: Consistent results across training and inference
- **⚡ High Performance**: Optimized for <50ms inference per sample
- **🌐 Cross-Platform**: Works in both Node.js and browser environments
- **📝 TypeScript First**: Full type safety with comprehensive JSDoc documentation
- **🧪 Production Ready**: Extensive testing, logging, and error handling

## 📦 Installation

```bash
npm install @your-org/decision-tree-model
```

## 🎯 Quick Start

### Basic Binary Classification

```typescript
import { Trainer, Model, FeatureAnalyzer } from '@your-org/decision-tree-model';

// Sample data
const data = [
  { brand: 'Apple', color: 'red', size: 'large', price: 100, sold: 'yes' },
  { brand: 'Samsung', color: 'blue', size: 'medium', price: 80, sold: 'no' },
  { brand: 'Apple', color: 'green', size: 'small', price: 60, sold: 'yes' },
  // ... more data
];

// 1. Analyze features
const analyzer = new FeatureAnalyzer();
const analysis = analyzer.analyzeFeatures(
  data,
  ['brand', 'color', 'size'], // categorical features
  ['price'],                  // numerical features
  'sold'                      // target variable
);

console.log('Feature analysis:', analysis);

// 2. Train model
const trainer = new Trainer({
  categoricalFeatures: ['brand', 'color', 'size'],
  numericFeatures: ['price'],
  target: 'sold',
  xgbParams: {
    objective: 'binary:logistic',
    max_depth: 6,
    eta: 0.3,
    num_boost_round: 100,
  },
});

await trainer.loadRaw(data);
await trainer.train({ useCrossValidation: true, nFolds: 5 });

// 3. Save model
await trainer.save('./models/sales_model');

// 4. Load and use model
const model = await Model.load('./models/sales_model');

// Make predictions
const prediction = model.predict({ 
  brand: 'Apple', 
  color: 'red', 
  size: 'large', 
  price: 95 
});

console.log('Prediction:', prediction); // 0 or 1 (class index)

// Get probabilities
const probabilities = model.predictProbabilities({ 
  brand: 'Apple', 
  color: 'red', 
  size: 'large', 
  price: 95 
});

console.log('Probabilities:', probabilities); // [0.2, 0.8] (probability for each class)
```

### Multi-class Classification

```typescript
import { Trainer, Model } from '@your-org/decision-tree-model';

const data = [
  { feature1: 'A', feature2: 'X', numeric1: 1.5, category: 'cat' },
  { feature1: 'B', feature2: 'Y', numeric1: 2.1, category: 'dog' },
  { feature1: 'A', feature2: 'Z', numeric1: 1.8, category: 'bird' },
  // ... more data
];

const trainer = new Trainer({
  categoricalFeatures: ['feature1', 'feature2'],
  numericFeatures: ['numeric1'],
  target: 'category',
  xgbParams: {
    objective: 'multi:softprob',
    max_depth: 4,
    eta: 0.1,
    num_boost_round: 50,
  },
});

await trainer.loadRaw(data);
await trainer.train();
await trainer.save('./models/multiclass_model');

const model = await Model.load('./models/multiclass_model');
const result = model.predict({ feature1: 'A', feature2: 'X', numeric1: 1.7 });
console.log('Predicted class:', result); // 0, 1, or 2 (class index)
```

### Loading Data from CSV

```typescript
import { Trainer } from '@your-org/decision-tree-model';

const trainer = new Trainer({
  categoricalFeatures: ['brand', 'category'],
  numericFeatures: ['price', 'rating'],
  target: 'sold',
  xgbParams: {
    objective: 'binary:logistic',
    max_depth: 6,
    eta: 0.3,
  },
});

// Load from CSV file
await trainer.loadCSV('./data/sales_data.csv');

// Train with cross-validation
await trainer.train({ 
  useCrossValidation: true, 
  nFolds: 5,
  testRatio: 0.2 
});

// Evaluate model performance
const metrics = trainer.evaluate();
console.log('Model metrics:', metrics);
```

## 🧩 Core Components

### 1. HashEncoder

Converts categorical features to numerical vectors using deterministic hashing:

```typescript
import { HashEncoder } from '@your-org/decision-tree-model';

const encoder = new HashEncoder(100, 42, 'color_feature');

// Encode categorical values
const encoded = encoder.encode('red');    // Float32Array with one-hot encoding
const batch = encoder.encodeBatch(['red', 'blue', 'green']);

// Check for deterministic behavior
const isDeterministic = encoder.validateDeterminism('red');
console.log('Is deterministic:', isDeterministic); // true
```

### 2. FeatureAnalyzer

Analyzes datasets to determine optimal feature specifications:

```typescript
import { FeatureAnalyzer } from '@your-org/decision-tree-model';

const analyzer = new FeatureAnalyzer();

// Auto-suggest feature types
const suggestions = analyzer.suggestFeatureTypes(data);
console.log('Categorical features:', suggestions.categorical);
console.log('Numerical features:', suggestions.numeric);

// Analyze feature correlations
const correlations = analyzer.analyzeCorrelation(data, ['price', 'rating']);
console.log('Feature correlations:', correlations);
```

### 3. Trainer

Handles the complete training pipeline:

```typescript
import { Trainer } from '@your-org/decision-tree-model';

const trainer = new Trainer({
  categoricalFeatures: ['brand', 'color'],
  numericFeatures: ['price'],
  target: 'sold',
  xgbParams: {
    objective: 'binary:logistic',
    max_depth: 6,
    eta: 0.3,
    num_boost_round: 100,
  },
});

// Load data from various sources
await trainer.loadCSV('./data.csv');
await trainer.loadJSON('./data.json');
await trainer.loadRaw(arrayOfObjects);

// Train with different configurations
await trainer.train({
  useCrossValidation: true,
  nFolds: 10,
  testRatio: 0.2,
  shuffle: true,
  seed: 42,
});

// Get training metrics
const metrics = trainer.evaluate();
console.log('Accuracy:', metrics.accuracy);
console.log('ROC AUC:', metrics.rocAuc);
console.log('Confusion Matrix:', metrics.confusionMatrix);
```

### 4. Model

Handles model loading and inference:

```typescript
import { Model } from '@your-org/decision-tree-model';

const model = await Model.load('./models/my_model');

// Single prediction
const prediction = model.predict({ feature1: 'value1', feature2: 123 });

// Batch prediction
const predictions = model.predictBatch([
  { feature1: 'value1', feature2: 123 },
  { feature1: 'value2', feature2: 456 },
]);

// Get prediction probabilities
const probabilities = model.predictProbabilities({ feature1: 'value1', feature2: 123 });

// Transform features without prediction
const transformed = model.transform({ feature1: 'value1', feature2: 123 });
```

## 🌳 Tree Visualization

The toolkit includes comprehensive tree visualization capabilities to help you understand and debug your XGBoost models. You can visualize decision trees in multiple formats: text, JSON, HTML, and SVG.

### Basic Usage

```typescript
import { createRegressionTrainer, Model } from '@your-org/decision-tree-model';

// Train a model
const trainer = createRegressionTrainer(['location', 'type'], ['area', 'age'], 'price');
await trainer.loadRaw(data);
await trainer.train();

// Visualize tree as text
const textViz = trainer.visualizeTree('text', {
  treeIndex: 0,
  includeFeatureNames: true,
  maxDepth: 3,
  precision: 2
});
console.log(textViz.content);

// Save HTML visualization
await trainer.saveTreeVisualization('./tree.html', 'html', {
  includeFeatureNames: true,
  maxDepth: 4
});
```

### Visualization Formats

#### Text Format
Perfect for quick debugging and console output:

```
Decision Tree
└── price <= 499.65
    ├── area <= 1200.50
    │   ├── Leaf: -0.25
    │   └── Leaf: 0.15
    └── Leaf: 0.75
```

#### JSON Format
Structured data for custom visualization tools:

```json
{
  "nodeId": "0",
  "depth": 0,
  "isLeaf": false,
  "featureIndex": 41,
  "featureName": "price",
  "threshold": 499.6452,
  "left": {
    "nodeId": "0.0",
    "depth": 1,
    "isLeaf": true,
    "value": -0.5
  },
  "right": {
    "nodeId": "0.1",
    "depth": 1,
    "isLeaf": true,
    "value": 0.5
  }
}
```

#### HTML Format
Interactive visualization with styling:
- Opens in web browsers
- Responsive design
- Color-coded nodes
- Tree statistics in header

#### SVG Format
Scalable vector graphics for high-quality output:
- Print-ready quality
- Embeddable in documents
- Programmatically generated
- Customizable styling

### Visualization Options

```typescript
interface TreeVisualizationOptions {
  /** Include feature names instead of indices */
  includeFeatureNames?: boolean;
  /** Maximum depth to display */
  maxDepth?: number;
  /** Include leaf values */
  includeValues?: boolean;
  /** Precision for floating point numbers */
  precision?: number;
  /** Tree index to visualize (for multi-tree models) */
  treeIndex?: number;
  /** Class name to visualize (for One-vs-Rest models) */
  className?: string;
}
```

### Advanced Usage

#### Tree Statistics
```typescript
const stats = trainer.getTreeStatistics();
console.log(`Model has ${stats.treeCount} trees`);
stats.trees.forEach(tree => {
  console.log(`Tree ${tree.index}: ${tree.nodeCount} nodes, depth ${tree.maxDepth}`);
});
```

#### Visualizing Loaded Models
```typescript
const model = await Model.load('./models/my_model');
const viz = model.visualizeTree('html', { maxDepth: 3 });
await model.saveTreeVisualization('./loaded_tree.svg', 'svg');
```

#### One-vs-Rest Models
```typescript
// For multi-class models using One-vs-Rest
const viz = trainer.visualizeTree('text', {
  className: 'class_A',  // Specify which class model to visualize
  includeFeatureNames: true
});
```

#### Direct TreeVisualizer Usage
```typescript
import { TreeVisualizer } from '@your-org/decision-tree-model';

const featureNames = ['feature1', 'feature2', 'feature3'];
const visualizer = new TreeVisualizer(featureNames);

// Get tree statistics
const stats = visualizer.getTreeStatistics(xgboostModel);

// Generate visualization
const result = visualizer.visualizeTree(xgboostModel, 'html', {
  maxDepth: 5,
  precision: 3
});
```

### Use Cases

1. **Model Understanding**: Visualize how your model makes decisions
2. **Debugging**: Identify problematic splits or overfitting
3. **Feature Importance**: See which features are used at each level
4. **Model Comparison**: Compare tree structures between different models
5. **Documentation**: Include tree visualizations in reports and presentations
6. **Education**: Teach others how decision trees work

### Best Practices

- Use `maxDepth` to focus on important parts of large trees
- Enable `includeFeatureNames` for better readability
- Use HTML format for interactive exploration
- Use SVG format for high-quality documentation
- Use JSON format for custom analysis tools
- Use text format for quick debugging

### Performance Tips

- Tree visualization is fast for small to medium trees
- For large trees (>1000 nodes), use `maxDepth` to limit visualization
- SVG generation is slower than other formats but produces high-quality output
- HTML files are self-contained and can be shared easily

## 🔧 Advanced Usage

### Custom XGBoost Parameters

```typescript
const trainer = new Trainer({
  categoricalFeatures: ['category'],
  numericFeatures: ['value'],
  target: 'label',
  xgbParams: {
    objective: 'binary:logistic',
    max_depth: 8,
    eta: 0.1,
    min_child_weight: 1,
    gamma: 0,
    subsample: 0.8,
    colsample_bytree: 0.8,
    alpha: 0,
    lambda: 1,
    seed: 42,
    num_boost_round: 200,
  },
});
```

### Feature Hashing Configuration

```typescript
import { HashEncoder } from '@your-org/decision-tree-model';

// Custom bucket count and seed
const encoder = new HashEncoder(
  1000,      // bucket count
  123,       // hash seed
  'feature'  // feature name
);

// Estimate collision probability
const collisionProb = HashEncoder.estimateCollisionProbability(500, 1000);
console.log('Collision probability:', collisionProb);

// Suggest optimal bucket count
const bucketCount = HashEncoder.suggestBucketCount(500, 0.05);
console.log('Suggested bucket count:', bucketCount);
```

### Cross-Validation and Model Evaluation

```typescript
await trainer.train({
  useCrossValidation: true,
  nFolds: 10,
  testRatio: 0.2,
  shuffle: true,
  seed: 42,
});

const metrics = trainer.evaluate();
console.log('Cross-validation scores:', metrics.crossValidationScores);
console.log('Mean CV score:', metrics.crossValidationScores?.reduce((a, b) => a + b, 0) / metrics.crossValidationScores?.length);
```

## 📊 Model Persistence

Models are saved with complete metadata for reproducibility:

```typescript
// Save model with metadata
await trainer.save('./models/my_model');

// This creates:
// - ./models/my_model/model.json (XGBoost model)
// - ./models/my_model/metadata.json (feature specs, classes, etc.)
// - ./models/my_model/encoders.json (encoder configurations)

// Load model
const model = await Model.load('./models/my_model');

// Check model metadata
const metadata = model.getMetadata();
console.log('Model version:', metadata.version);
console.log('Training date:', metadata.createdAt);
console.log('Features:', metadata.categoricalFeatures, metadata.numericFeatures);
```

## 🧪 Testing and Validation

### Unit Testing

```typescript
import { HashEncoder } from '@your-org/decision-tree-model';

describe('HashEncoder', () => {
  test('should produce deterministic encodings', () => {
    const encoder = new HashEncoder(100, 42);
    const encoding1 = encoder.encode('test');
    const encoding2 = encoder.encode('test');
    expect(encoding1).toEqual(encoding2);
  });

  test('should handle null values', () => {
    const encoder = new HashEncoder(100, 42);
    const encoding = encoder.encode(null);
    expect(encoding).toBeInstanceOf(Float32Array);
    expect(encoding.length).toBe(100);
  });
});
```

### Model Validation

```typescript
// Validate model consistency
const model = await Model.load('./models/my_model');
const testData = { feature1: 'test', feature2: 123 };

const prediction1 = model.predict(testData);
const prediction2 = model.predict(testData);
console.log('Consistent predictions:', prediction1 === prediction2);

// Test with unseen categories
const unseenData = { feature1: 'never_seen_before', feature2: 456 };
const prediction = model.predict(unseenData); // Should work gracefully
```

## 🚀 Performance Optimization

### Batch Processing

```typescript
// Process multiple samples at once
const batchData = [
  { feature1: 'A', feature2: 123 },
  { feature1: 'B', feature2: 456 },
  { feature1: 'C', feature2: 789 },
];

const batchPredictions = model.predictBatch(batchData);
console.log('Batch predictions:', batchPredictions);
```

### Memory Management

```typescript
// For large datasets, consider streaming
import { readFileSync } from 'fs';
import { parse } from 'papaparse';

const csvData = readFileSync('./large_dataset.csv', 'utf8');
const parsed = parse(csvData, { 
  header: true, 
  dynamicTyping: true,
  skipEmptyLines: true 
});

// Process in chunks
const chunkSize = 10000;
for (let i = 0; i < parsed.data.length; i += chunkSize) {
  const chunk = parsed.data.slice(i, i + chunkSize);
  // Process chunk...
}
```

## 🔍 Debugging and Monitoring

### Logging

```typescript
import { Trainer, Logger, LogLevel } from '@your-org/decision-tree-model';

// Custom logger
const logger: Logger = {
  debug: (msg, meta) => console.debug(`[DEBUG] ${msg}`, meta),
  info: (msg, meta) => console.info(`[INFO] ${msg}`, meta),
  warn: (msg, meta) => console.warn(`[WARN] ${msg}`, meta),
  error: (msg, meta) => console.error(`[ERROR] ${msg}`, meta),
};

const trainer = new Trainer({
  categoricalFeatures: ['brand'],
  numericFeatures: ['price'],
  target: 'sold',
  xgbParams: { objective: 'binary:logistic' },
}, logger);
```

### Error Handling

```typescript
try {
  const trainer = new Trainer({
    categoricalFeatures: ['brand'],
    numericFeatures: ['price'],
    target: 'sold',
    xgbParams: { objective: 'binary:logistic' },
  });

  await trainer.loadCSV('./data.csv');
  await trainer.train();
  await trainer.save('./models/my_model');
} catch (error) {
  console.error('Training failed:', error.message);
  // Handle specific error types
  if (error.message.includes('Features not found')) {
    console.error('Check your feature names in the data');
  }
}
```

## 📚 API Reference

### Main Classes

- **`Trainer`**: Main class for training models
- **`Model`**: Class for loading and using trained models
- **`HashEncoder`**: Categorical feature encoder
- **`FeatureAnalyzer`**: Dataset analysis utility

### Utilities

- **`Utils`**: Data manipulation utilities
- **`Metrics`**: Model evaluation metrics
- **`DataLoader`**: Data loading helpers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run tests: `npm test`
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Documentation](https://your-org.github.io/decision-tree-model/)
- [API Reference](https://your-org.github.io/decision-tree-model/api/)
- [Examples](https://github.com/your-org/decision-tree-model/tree/main/examples)
- [Issues](https://github.com/your-org/decision-tree-model/issues)

## 📈 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

Made with ❤️ by the Decision Tree Model team 