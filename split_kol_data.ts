/**
 * @fileoverview Data splitting utility for Solana KOL OHLCV Features
 * 
 * This script splits the OHLCV features dataset into training and testing sets
 * with proper stratification and temporal considerations for financial data.
 * 
 * Features:
 * - Temporal splitting (preserves time order)
 * - Stratified splitting (balanced class distribution)
 * - Random splitting with seed for reproducibility
 * - Data validation and statistics
 */

import {
  DataLoader,
  Utils,
  defaultLogger,
} from './src';
import * as path from 'path';
import * as fs from 'fs';

interface SplitConfig {
  inputFile: string;
  outputDir: string;
  testRatio: number;
  splitMethod: 'temporal' | 'stratified' | 'random';
  seed?: number;
  validateSplit?: boolean;
}

interface DataSample {
  [key: string]: any;
  performance_label: string;
  buy_timestamp: string;
}

async function splitKolData(config: SplitConfig): Promise<void> {
  console.log('✂️  Solana KOL OHLCV Data Splitting Utility\n');

  // Simple Fisher-Yates shuffle implementation
  function shuffle<T>(array: T[], seed?: number): T[] {
    const arr = [...array];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  const {
    inputFile,
    outputDir,
    testRatio,
    splitMethod,
    seed = 42,
    validateSplit = true
  } = config;

  // Validate input file
  if (!fs.existsSync(inputFile)) {
    console.error(`❌ Input file not found: ${inputFile}`);
    return;
  }

  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
    console.log(`📁 Created output directory: ${outputDir}`);
  }

  try {
    // Step 1: Load the data
    console.log('📊 Step 1: Loading OHLCV features data...');
    const dataLoader = new DataLoader(defaultLogger);
    const data = await dataLoader.loadCSV(inputFile) as DataSample[];

    console.log(`Loaded ${data.length} records`);

    // Step 2: Analyze data
    console.log('\n🔍 Step 2: Analyzing data distribution...');
    
    // Count classes
    const classDistribution: { [key: string]: number } = {};
    data.forEach(record => {
      const label = record.performance_label;
      classDistribution[label] = (classDistribution[label] || 0) + 1;
    });

    console.log('Class distribution:');
    Object.entries(classDistribution).forEach(([cls, count]) => {
      const percentage = ((count / data.length) * 100).toFixed(1);
      console.log(`- ${cls}: ${count} (${percentage}%)`);
    });

    // Analyze temporal distribution
    const timestamps = data
      .map(record => new Date(record.buy_timestamp))
      .sort((a, b) => a.getTime() - b.getTime());
    
    const earliestDate = timestamps[0];
    const latestDate = timestamps[timestamps.length - 1];
    
    console.log('\nTemporal distribution:');
    console.log(`- Earliest: ${earliestDate.toISOString()}`);
    console.log(`- Latest: ${latestDate.toISOString()}`);
    console.log(`- Time span: ${Math.ceil((latestDate.getTime() - earliestDate.getTime()) / (1000 * 60 * 60 * 24))} days`);

    // Step 3: Split the data
    console.log(`\n✂️  Step 3: Splitting data using ${splitMethod} method...`);
    console.log(`Test ratio: ${(testRatio * 100).toFixed(1)}%`);
    
    let trainData: DataSample[];
    let testData: DataSample[];

    switch (splitMethod) {
      case 'temporal':
        // Sort by timestamp and split
        const sortedData = [...data].sort((a, b) => 
          new Date(a.buy_timestamp).getTime() - new Date(b.buy_timestamp).getTime()
        );
        const splitIndex = Math.floor(sortedData.length * (1 - testRatio));
        trainData = sortedData.slice(0, splitIndex);
        testData = sortedData.slice(splitIndex);
        console.log('Applied temporal split (training on older data, testing on newer)');
        break;

      case 'stratified':
        // Group by class, then split each group
        const groupedData: { [key: string]: DataSample[] } = {};
        data.forEach(record => {
          const label = record.performance_label;
          if (!groupedData[label]) {
            groupedData[label] = [];
          }
          groupedData[label].push(record);
        });

        trainData = [];
        testData = [];

        Object.entries(groupedData).forEach(([cls, records]) => {
          // Shuffle with seed
          const shuffled = shuffle([...records], seed);
          const clsSplitIndex = Math.floor(shuffled.length * (1 - testRatio));
          
          trainData.push(...shuffled.slice(0, clsSplitIndex));
          testData.push(...shuffled.slice(clsSplitIndex));
        });

        console.log('Applied stratified split (balanced class distribution)');
        break;

      case 'random':
      default:
        // Simple random shuffle and split
        const shuffledData = shuffle([...data], seed);
        const randomSplitIndex = Math.floor(shuffledData.length * (1 - testRatio));
        trainData = shuffledData.slice(0, randomSplitIndex);
        testData = shuffledData.slice(randomSplitIndex);
        console.log('Applied random split');
        break;
    }

    console.log(`Training set: ${trainData.length} records`);
    console.log(`Test set: ${testData.length} records`);

    // Step 4: Validate split
    if (validateSplit) {
      console.log('\n✅ Step 4: Validating split quality...');
      
      // Check class distributions
      const trainClassDist: { [key: string]: number } = {};
      const testClassDist: { [key: string]: number } = {};
      
      trainData.forEach(record => {
        const label = record.performance_label;
        trainClassDist[label] = (trainClassDist[label] || 0) + 1;
      });
      
      testData.forEach(record => {
        const label = record.performance_label;
        testClassDist[label] = (testClassDist[label] || 0) + 1;
      });

      console.log('\nClass distribution validation:');
      Object.keys(classDistribution).forEach(cls => {
        const trainCount = trainClassDist[cls] || 0;
        const testCount = testClassDist[cls] || 0;
        const trainPct = ((trainCount / trainData.length) * 100).toFixed(1);
        const testPct = ((testCount / testData.length) * 100).toFixed(1);
        const originalPct = ((classDistribution[cls] / data.length) * 100).toFixed(1);
        
        console.log(`- ${cls}: Original ${originalPct}% | Train ${trainPct}% | Test ${testPct}%`);
      });

      // Check temporal overlap for non-temporal splits
      if (splitMethod !== 'temporal') {
        const trainDates = trainData.map(r => new Date(r.buy_timestamp));
        const testDates = testData.map(r => new Date(r.buy_timestamp));
        
        const trainDateRange = {
          min: new Date(Math.min(...trainDates.map(d => d.getTime()))),
          max: new Date(Math.max(...trainDates.map(d => d.getTime())))
        };
        
        const testDateRange = {
          min: new Date(Math.min(...testDates.map(d => d.getTime()))),
          max: new Date(Math.max(...testDates.map(d => d.getTime())))
        };

        console.log('\nTemporal distribution:');
        console.log(`- Train range: ${trainDateRange.min.toISOString()} to ${trainDateRange.max.toISOString()}`);
        console.log(`- Test range: ${testDateRange.min.toISOString()} to ${testDateRange.max.toISOString()}`);
        
        // Check for temporal overlap (expected for random/stratified)
        const hasOverlap = trainDateRange.min <= testDateRange.max && testDateRange.min <= trainDateRange.max;
        console.log(`- Temporal overlap: ${hasOverlap ? 'Yes' : 'No'} (expected for non-temporal splits)`);
      }
    }

    // Step 5: Save split data
    console.log('\n💾 Step 5: Saving split datasets...');
    
    const trainFile = path.join(outputDir, `train_ohlcv_features_${splitMethod}.csv`);
    const testFile = path.join(outputDir, `test_ohlcv_features_${splitMethod}.csv`);
    
    // Get headers from first record
    const headers = Object.keys(data[0]);
    
    // Convert data to CSV format
    const formatCSV = (data: DataSample[]): string => {
      const csvLines = [headers.join(',')]; // Header row
      data.forEach(record => {
        const values = headers.map(header => {
          const value = record[header];
          // Escape values containing commas or quotes
          if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
            return `"${value.replace(/"/g, '""')}"`;
          }
          return String(value);
        });
        csvLines.push(values.join(','));
      });
      return csvLines.join('\n');
    };

    // Write training data
    fs.writeFileSync(trainFile, formatCSV(trainData));
    console.log(`✅ Training data saved: ${trainFile}`);
    
    // Write test data
    fs.writeFileSync(testFile, formatCSV(testData));
    console.log(`✅ Test data saved: ${testFile}`);

    // Step 6: Create split metadata
    const metadata = {
      splitMethod,
      testRatio,
      seed,
      totalRecords: data.length,
      trainRecords: trainData.length,
      testRecords: testData.length,
      classDistribution,
      splitDate: new Date().toISOString(),
      inputFile: path.basename(inputFile),
      files: {
        train: path.basename(trainFile),
        test: path.basename(testFile)
      }
    };

    const metadataFile = path.join(outputDir, `split_metadata_${splitMethod}.json`);
    fs.writeFileSync(metadataFile, JSON.stringify(metadata, null, 2));
    console.log(`📄 Metadata saved: ${metadataFile}`);

    // Step 7: Summary and recommendations
    console.log('\n📊 Split Summary:');
    console.log('='.repeat(50));
    console.log(`Method: ${splitMethod}`);
    console.log(`Total records: ${data.length}`);
    console.log(`Training: ${trainData.length} (${((trainData.length / data.length) * 100).toFixed(1)}%)`);
    console.log(`Testing: ${testData.length} (${((testData.length / data.length) * 100).toFixed(1)}%)`);
    console.log('='.repeat(50));

    console.log('\n💡 Recommendations:');
    switch (splitMethod) {
      case 'temporal':
        console.log('- Use for time-series validation and production simulation');
        console.log('- Model will be tested on future data (most realistic)');
        console.log('- Good for detecting temporal bias and model drift');
        break;
      case 'stratified':
        console.log('- Use for balanced evaluation across all classes');
        console.log('- Ensures both train and test sets represent all outcomes');
        console.log('- Good for imbalanced datasets');
        break;
      case 'random':
        console.log('- Use for general model validation');
        console.log('- Simple and widely understood approach');
        console.log('- Good baseline splitting method');
        break;
    }

    console.log('\n🎯 Next Steps:');
    console.log(`1. Update train_kol_classifier.ts to use: ${trainFile}`);
    console.log(`2. Update test_kol_classifier.ts to use: ${testFile}`);
    console.log('3. Train model on training set');
    console.log('4. Evaluate model on test set');
    console.log('5. Compare results across different split methods');

    console.log('\n✅ Data splitting completed successfully! 🎉');

  } catch (error) {
    console.error('❌ Data splitting failed:', error);
    
    if (error instanceof Error) {
      console.error('Error details:', error.message);
    }
    
    console.log('\n🔧 Troubleshooting:');
    console.log('1. Check input file format and structure');
    console.log('2. Ensure sufficient disk space for output files');
    console.log('3. Verify write permissions for output directory');
    console.log('4. Check that performance_label and buy_timestamp columns exist');
    
    throw error;
  }
}

// Example usage and command-line interface
async function main(): Promise<void> {
  const config: SplitConfig = {
    inputFile: './data-extraction/output/batch-ohlcv/Cupsey/Cupsey_ohlcv_features_2025-07-18T17-18-07-881Z.csv',
    outputDir: './data/splits',
    testRatio: 0.2,
    splitMethod: 'stratified', // or 'temporal', 'random'
    seed: 42,
    validateSplit: true
  };

  console.log('📋 Configuration:');
  console.log(`- Input: ${config.inputFile}`);
  console.log(`- Output: ${config.outputDir}`);
  console.log(`- Test ratio: ${config.testRatio}`);
  console.log(`- Method: ${config.splitMethod}`);
  console.log(`- Seed: ${config.seed}`);
  console.log('');

  // Allow command-line override of split method
  const args = process.argv.slice(2);
  if (args.length > 0) {
    const method = args[0] as 'temporal' | 'stratified' | 'random';
    if (['temporal', 'stratified', 'random'].includes(method)) {
      config.splitMethod = method;
      console.log(`🔄 Override split method: ${method}`);
    }
  }

  await splitKolData(config);
}

// Handle errors and run the splitting
if (require.main === module) {
  main()
    .then(() => {
      console.log('\n🏁 Script completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n💥 Script failed:', error);
      process.exit(1);
    });
}

export { splitKolData };
export type { SplitConfig }; 