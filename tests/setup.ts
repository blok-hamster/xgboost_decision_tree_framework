/**
 * @fileoverview Jest test setup configuration
 * 
 * This file is executed before each test file to set up the testing environment.
 */

// Mock console methods to avoid noise in test output
global.console = {
  ...console,
  // Keep important methods
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Set up test environment variables
process.env.NODE_ENV = 'test';

// Mock performance timing for tests
global.performance = {
  now: jest.fn(() => Date.now()),
} as any;

// Mock file system operations for browser environment tests
if (typeof window !== 'undefined') {
  // Browser environment mocks
  global.fs = {
    readFileSync: jest.fn(),
    writeFileSync: jest.fn(),
    existsSync: jest.fn(),
    statSync: jest.fn(),
  };
}

// Set up global test helpers
global.expectToBeCloseToArray = (actual: number[], expected: number[], precision: number = 5) => {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], precision);
  }
};

// Set up common test data
global.testData = {
  simpleBinary: [
    { feature1: 'A', feature2: 'X', numeric1: 1.0, target: 'yes' },
    { feature1: 'B', feature2: 'Y', numeric1: 2.0, target: 'no' },
    { feature1: 'A', feature2: 'Z', numeric1: 1.5, target: 'yes' },
    { feature1: 'C', feature2: 'X', numeric1: 3.0, target: 'no' },
  ],
  simpleMulticlass: [
    { feature1: 'A', feature2: 'X', numeric1: 1.0, target: 'cat' },
    { feature1: 'B', feature2: 'Y', numeric1: 2.0, target: 'dog' },
    { feature1: 'A', feature2: 'Z', numeric1: 1.5, target: 'bird' },
    { feature1: 'C', feature2: 'X', numeric1: 3.0, target: 'cat' },
  ],
  withNulls: [
    { feature1: 'A', feature2: null, numeric1: 1.0, target: 'yes' },
    { feature1: null, feature2: 'Y', numeric1: 2.0, target: 'no' },
    { feature1: 'A', feature2: 'Z', numeric1: null, target: 'yes' },
    { feature1: 'C', feature2: 'X', numeric1: 3.0, target: null },
  ],
};

// Export test utilities
export {}; 