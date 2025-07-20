/**
 * TypeScript declarations for @fractal-solutions/xgboost-js
 */

declare module '@fractal-solutions/xgboost-js' {
  interface XGBoostParams {
    learningRate?: number;
    maxDepth?: number;
    minChildWeight?: number;
    numRounds?: number;
    [key: string]: any;
  }

  interface XGBoostJSON {
    trees: any[];
    params: XGBoostParams;
  }

  export class XGBoost {
    constructor(params?: XGBoostParams);
    
    fit(X: number[][], y: number[]): void;
    predictSingle(x: number[]): number;
    predictBatch(X: number[][]): number[];
    getFeatureImportance(): number[];
    toJSON(): XGBoostJSON;
    static fromJSON(json: XGBoostJSON): XGBoost;
  }

  export default XGBoost;
} 