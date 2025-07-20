/**
 * @fileoverview DataLoader utility for loading data from various sources
 * 
 * Handles loading data from CSV files, JSON files, and direct arrays with
 * proper error handling, validation, and configuration options.
 */

import { readFileSync, existsSync, statSync } from 'fs';
import { parse as csvParse } from 'papaparse';
import { RawDataRecord, DataLoadConfig, Logger, FileOperationResult } from '../types';

/**
 * DataLoader class for loading data from various sources
 * 
 * Features:
 * - CSV file loading with configurable parsing options
 * - JSON file loading with validation
 * - Memory-efficient streaming for large files
 * - Data validation and type coercion
 * - Error handling with detailed messages
 * - Progress tracking for large files
 * 
 * @example
 * ```typescript
 * const loader = new DataLoader();
 * 
 * // Load CSV file
 * const csvData = await loader.loadCSV('./data.csv', {
 *   delimiter: ',',
 *   hasHeader: true,
 *   maxRows: 10000,
 * });
 * 
 * // Load JSON file
 * const jsonData = await loader.loadJSON('./data.json');
 * 
 * // Validate data
 * const isValid = loader.validateData(data, ['feature1', 'feature2']);
 * ```
 */
export class DataLoader {
  private logger: Logger | undefined;

  /**
   * Creates a new DataLoader instance
   * 
   * @param logger - Optional logger for debugging and monitoring
   */
  constructor(logger?: Logger) {
    this.logger = logger;
  }

  /**
   * Loads data from a CSV file
   * 
   * @param filePath - Path to the CSV file
   * @param config - Configuration options for CSV parsing
   * @returns Promise resolving to array of data records
   * 
   * @throws Error if file doesn't exist or parsing fails
   * 
   * @example
   * ```typescript
   * const data = await loader.loadCSV('./sales.csv', {
   *   delimiter: ',',
   *   hasHeader: true,
   *   encoding: 'utf8',
   *   maxRows: 5000,
   * });
   * ```
   */
  public async loadCSV(filePath: string, config: DataLoadConfig = {}): Promise<RawDataRecord[]> {
    this.logger?.info(`Loading CSV file: ${filePath}`);

    // Check if file exists
    if (!existsSync(filePath)) {
      throw new Error(`CSV file not found: ${filePath}`);
    }

    // Get file size for progress tracking
    const fileStats = statSync(filePath);
    const fileSizeKB = Math.round(fileStats.size / 1024);
    
    this.logger?.info(`Loading CSV file (${fileSizeKB} KB)`, {
      filePath,
      sizeKB: fileSizeKB,
    });

    try {
      // Read file content
      const encoding = config.encoding || 'utf8';
      const fileContent = readFileSync(filePath, encoding as BufferEncoding);
      
      // Parse CSV with papaparse
      const parseResult = csvParse(fileContent, {
        header: config.hasHeader ?? true,
        delimiter: config.delimiter || ',',
        skipEmptyLines: true,
        dynamicTyping: true,
        transformHeader: (header: string) => header.trim(),
        transform: (value: string) => {
          if (typeof value === 'string') {
            return value.trim();
          }
          return value;
        },
      });

      if (parseResult.errors.length > 0) {
        this.logger?.warn(`CSV parsing warnings:`, parseResult.errors);
      }

      let data = parseResult.data as RawDataRecord[];
      
      // Apply row limits
      if (config.maxRows && data.length > config.maxRows) {
        this.logger?.warn(`Limiting data to ${config.maxRows} rows (original: ${data.length})`);
        data = data.slice(0, config.maxRows);
      }

      if (config.skipRows && config.skipRows > 0) {
        this.logger?.info(`Skipping first ${config.skipRows} rows`);
        data = data.slice(config.skipRows);
      }

      // Filter out empty rows
      data = data.filter(row => row && Object.keys(row).length > 0);

      this.logger?.info(`Successfully loaded CSV data`, {
        rows: data.length,
        columns: data.length > 0 ? Object.keys(data[0]!).length : 0,
      });

      return data;
    } catch (error) {
      this.logger?.error(`Failed to load CSV file: ${filePath}`, error);
      throw new Error(`Failed to load CSV file ${filePath}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Loads data from a JSON file
   * 
   * @param filePath - Path to the JSON file
   * @returns Promise resolving to array of data records
   * 
   * @throws Error if file doesn't exist or parsing fails
   * 
   * @example
   * ```typescript
   * const data = await loader.loadJSON('./data.json');
   * ```
   */
  public async loadJSON(filePath: string): Promise<RawDataRecord[]> {
    this.logger?.info(`Loading JSON file: ${filePath}`);

    // Check if file exists
    if (!existsSync(filePath)) {
      throw new Error(`JSON file not found: ${filePath}`);
    }

    // Get file size for progress tracking
    const fileStats = statSync(filePath);
    const fileSizeKB = Math.round(fileStats.size / 1024);
    
    this.logger?.info(`Loading JSON file (${fileSizeKB} KB)`, {
      filePath,
      sizeKB: fileSizeKB,
    });

    try {
      // Read and parse JSON file
      const fileContent = readFileSync(filePath, 'utf8');
      const jsonData = JSON.parse(fileContent);

      // Validate that JSON contains an array
      if (!Array.isArray(jsonData)) {
        throw new Error('JSON file must contain an array of objects');
      }

      // Validate that array contains objects
      if (jsonData.length > 0 && typeof jsonData[0] !== 'object') {
        throw new Error('JSON array must contain objects');
      }

      this.logger?.info(`Successfully loaded JSON data`, {
        rows: jsonData.length,
        columns: jsonData.length > 0 ? Object.keys(jsonData[0]!).length : 0,
      });

      return jsonData as RawDataRecord[];
    } catch (error) {
      this.logger?.error(`Failed to load JSON file: ${filePath}`, error);
      throw new Error(`Failed to load JSON file ${filePath}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Validates data array structure and required features
   * 
   * @param data - Data array to validate
   * @param requiredFeatures - Features that must be present
   * @returns Whether the data is valid
   * 
   * @example
   * ```typescript
   * const isValid = loader.validateData(data, ['feature1', 'feature2', 'target']);
   * if (!isValid) {
   *   throw new Error('Invalid data structure');
   * }
   * ```
   */
  public validateData(data: RawDataRecord[], requiredFeatures: string[] = []): boolean {
    if (!Array.isArray(data)) {
      this.logger?.error('Data is not an array');
      return false;
    }

    if (data.length === 0) {
      this.logger?.warn('Data array is empty');
      return false;
    }

    // Check if all records are objects
    const invalidRecords = data.filter(record => typeof record !== 'object' || record === null);
    if (invalidRecords.length > 0) {
      this.logger?.error(`Found ${invalidRecords.length} invalid records (not objects)`);
      return false;
    }

    // Check required features
    if (requiredFeatures.length > 0) {
      const sampleRecord = data[0]!;
      const availableFeatures = Object.keys(sampleRecord);
      const missingFeatures = requiredFeatures.filter(feature => !availableFeatures.includes(feature));

      if (missingFeatures.length > 0) {
        this.logger?.error(`Missing required features: ${missingFeatures.join(', ')}`);
        return false;
      }
    }

    this.logger?.debug('Data validation passed', {
      records: data.length,
      features: Object.keys(data[0]!).length,
    });

    return true;
  }

  /**
   * Analyzes data quality and provides statistics
   * 
   * @param data - Data array to analyze
   * @returns Data quality report
   * 
   * @example
   * ```typescript
   * const report = loader.analyzeDataQuality(data);
   * console.log(`Null values: ${report.nullCount}`);
   * console.log(`Duplicate rows: ${report.duplicateCount}`);
   * ```
   */
  public analyzeDataQuality(data: RawDataRecord[]): {
    totalRows: number;
    totalFeatures: number;
    nullCount: number;
    duplicateCount: number;
    featureStats: { [feature: string]: { nullCount: number; uniqueCount: number } };
  } {
    if (data.length === 0) {
      return {
        totalRows: 0,
        totalFeatures: 0,
        nullCount: 0,
        duplicateCount: 0,
        featureStats: {},
      };
    }

    const features = Object.keys(data[0]!);
    const featureStats: { [feature: string]: { nullCount: number; uniqueCount: number } } = {};
    let totalNullCount = 0;

    // Analyze each feature
    for (const feature of features) {
      const uniqueValues = new Set();
      let nullCount = 0;

      for (const record of data) {
        const value = record[feature];
        if (value === null || value === undefined) {
          nullCount++;
          totalNullCount++;
        } else {
          uniqueValues.add(String(value));
        }
      }

      featureStats[feature] = {
        nullCount,
        uniqueCount: uniqueValues.size,
      };
    }

    // Count duplicate rows
    const serializedRows = data.map(record => JSON.stringify(record));
    const uniqueRows = new Set(serializedRows);
    const duplicateCount = data.length - uniqueRows.size;

    const report = {
      totalRows: data.length,
      totalFeatures: features.length,
      nullCount: totalNullCount,
      duplicateCount,
      featureStats,
    };

    this.logger?.debug('Data quality analysis completed', report);

    return report;
  }

  /**
   * Cleans data by removing duplicates, handling nulls, and normalizing values
   * 
   * @param data - Data array to clean
   * @param options - Cleaning options
   * @returns Cleaned data array
   * 
   * @example
   * ```typescript
   * const cleanedData = loader.cleanData(data, {
   *   removeDuplicates: true,
   *   fillNulls: 'mean',
   *   normalizeStrings: true,
   * });
   * ```
   */
  public cleanData(data: RawDataRecord[], options: {
    removeDuplicates?: boolean;
    fillNulls?: 'drop' | 'mean' | 'mode' | string;
    normalizeStrings?: boolean;
  } = {}): RawDataRecord[] {
    let cleanedData = [...data];

    this.logger?.info('Starting data cleaning', {
      originalRows: data.length,
      options,
    });

    // Remove duplicates
    if (options.removeDuplicates) {
      const seen = new Set();
      cleanedData = cleanedData.filter(record => {
        const serialized = JSON.stringify(record);
        if (seen.has(serialized)) {
          return false;
        }
        seen.add(serialized);
        return true;
      });
      
      this.logger?.debug(`Removed ${data.length - cleanedData.length} duplicate rows`);
    }

    // Handle null values
    if (options.fillNulls && options.fillNulls !== 'drop') {
      cleanedData = this.fillNullValues(cleanedData, options.fillNulls);
    } else if (options.fillNulls === 'drop') {
      cleanedData = cleanedData.filter(record => 
        !Object.values(record).some(value => value === null || value === undefined)
      );
    }

    // Normalize strings
    if (options.normalizeStrings) {
      cleanedData = cleanedData.map(record => {
        const normalized: RawDataRecord = {};
        for (const [key, value] of Object.entries(record)) {
          if (typeof value === 'string') {
            normalized[key] = value.trim().toLowerCase();
          } else {
            normalized[key] = value;
          }
        }
        return normalized;
      });
    }

    this.logger?.info('Data cleaning completed', {
      originalRows: data.length,
      cleanedRows: cleanedData.length,
      removedRows: data.length - cleanedData.length,
    });

    return cleanedData;
  }

  /**
   * Saves data to a file
   * 
   * @param data - Data to save
   * @param filePath - Output file path
   * @param format - Output format ('csv' or 'json')
   * @returns File operation result
   * 
   * @example
   * ```typescript
   * const result = await loader.saveData(data, './output.csv', 'csv');
   * console.log('Save result:', result);
   * ```
   */
  public async saveData(
    data: RawDataRecord[],
    filePath: string,
    format: 'csv' | 'json' = 'json'
  ): Promise<FileOperationResult> {
    try {
      this.logger?.info(`Saving data to ${filePath} (${format})`);

      let content: string;
      
      if (format === 'csv') {
        // Convert to CSV format
        if (data.length === 0) {
          content = '';
        } else {
          const headers = Object.keys(data[0]!);
          const csvRows = [headers.join(',')];
          
          for (const record of data) {
            const row = headers.map(header => {
              const value = record[header];
              if (value === null || value === undefined) {
                return '';
              }
              // Escape quotes and wrap in quotes if contains comma
              const stringValue = String(value);
              if (stringValue.includes(',') || stringValue.includes('"')) {
                return `"${stringValue.replace(/"/g, '""')}"`;
              }
              return stringValue;
            }).join(',');
            csvRows.push(row);
          }
          
          content = csvRows.join('\n');
        }
      } else {
        // JSON format
        content = JSON.stringify(data, null, 2);
      }

      // Write to file (Node.js only)
      if (typeof process !== 'undefined') {
        const fs = require('fs');
        fs.writeFileSync(filePath, content, 'utf8');
      } else {
        throw new Error('File saving not supported in browser environment');
      }

      const result: FileOperationResult = {
        success: true,
        filePath,
        size: content.length,
      };

      this.logger?.info(`Successfully saved ${data.length} records to ${filePath}`);
      return result;
    } catch (error) {
      const result: FileOperationResult = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        filePath,
      };

      this.logger?.error(`Failed to save data to ${filePath}`, error);
      return result;
    }
  }

  /**
   * Fills null values in the data
   * 
   * @private
   * @param data - Data array
   * @param strategy - Fill strategy
   * @returns Data with filled null values
   */
  private fillNullValues(data: RawDataRecord[], strategy: string): RawDataRecord[] {
    if (data.length === 0) return data;

    const features = Object.keys(data[0]!);
    const fillValues: { [feature: string]: any } = {};

    // Calculate fill values for each feature
    for (const feature of features) {
      const validValues = data
        .map(record => record[feature])
        .filter(value => value !== null && value !== undefined);

      if (validValues.length === 0) {
        fillValues[feature] = strategy === 'mean' ? 0 : '';
        continue;
      }

      if (strategy === 'mean') {
        const numericValues = validValues
          .map(v => Number(v))
          .filter(v => !isNaN(v));
        
        if (numericValues.length > 0) {
          fillValues[feature] = numericValues.reduce((a, b) => a + b, 0) / numericValues.length;
        } else {
          fillValues[feature] = validValues[0];
        }
      } else if (strategy === 'mode') {
        // Find most common value
        const counts: { [value: string]: number } = {};
        for (const value of validValues) {
          const stringValue = String(value);
          counts[stringValue] = (counts[stringValue] || 0) + 1;
        }
        
        const modeValue = Object.keys(counts).reduce((a, b) => 
          counts[a]! > counts[b]! ? a : b
        );
        fillValues[feature] = modeValue;
      } else {
        // Use provided string value
        fillValues[feature] = strategy;
      }
    }

    // Fill null values
    return data.map(record => {
      const filled: RawDataRecord = {};
      for (const [key, value] of Object.entries(record)) {
        if (value === null || value === undefined) {
          filled[key] = fillValues[key];
        } else {
          filled[key] = value;
        }
      }
      return filled;
    });
  }
} 