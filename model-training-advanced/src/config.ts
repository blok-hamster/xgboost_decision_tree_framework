
import path from 'path';

export const CONFIG = {
  // Path to the 'token-positions-*.json' files
  ANALYSIS_DIR: path.resolve(__dirname, '../../data-extraction/output/analysis'),
  
  // Path to the OHLCV CSV files
  OHLCV_DIR: path.resolve(__dirname, '../../data-extraction/output/ohlcv'),
  
  // Output path for the training data
  OUTPUT_FILE: path.resolve(__dirname, '../training_data_advanced.csv')
};
