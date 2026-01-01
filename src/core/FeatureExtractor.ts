
import { Candle, AdvancedFeatures } from '../types';

/**
 * FeatureExtractor class
 * 
 * Provides static methods to extract advanced OHLCV features from raw candle data.
 * Designed to be used at runtime with the trained Decision Tree model.
 * 
 * Handles "Dynamic Lookback" to support tokens with limited history (e.g., < 20 candles).
 */
export class FeatureExtractor {

  /**
   * Calculates all advanced features for the LATEST candle in the series.
   * Logic adapts to available history length (Dynamic Lookback).
   * 
   * @param candles - Array of OHLCV candles, sorted by timestamp ascending
   * @returns AdvancedFeatures object ready for model inference
   * @throws Error if candle array is empty
   */
  public static extract(candles: Candle[]): AdvancedFeatures {
    if (!candles || candles.length === 0) {
      throw new Error("Cannot extract features from empty candle array");
    }

    const lastCandle = candles[candles.length - 1]!;
    const firstCandle = candles[0]!;

    // Calculate Visible Token Age (in Hours)
    // Formula: (Last Timestamp - First Timestamp) / 3600
    // Uses the actual data span available
    const ageSeconds = lastCandle.timestamp - firstCandle.timestamp;
    const ageHours = ageSeconds / 3600;

    return {
      rvol: this.calculateRVOL(candles),
      k_percentage: this.calculateStochastic(candles, 'K'),
      d_percentage: this.calculateStochastic(candles, 'D'),
      natr: this.calculateNATR(candles),
      vwap_deviation: this.calculateVWAPDeviation(candles),
      upper_wick_ratio: this.calculateWickRatio(lastCandle, 'upper'),
      lower_wick_ratio: this.calculateWickRatio(lastCandle, 'lower'),
      market_cap: lastCandle.market_cap || 0,
      visible_token_age: ageHours < 0 ? 0 : ageHours // Ensure non-negative
    };
  }

  /**
   * Relative Volume (RVOL)
   * Formula: Current Volume / SMA(Volume, 20)
   * If available history < 20, uses SMA(Volume, available_history)
   */
  private static calculateRVOL(candles: Candle[]): number {
    const period = 20;
    const currentVol = candles[candles.length - 1]!.volume;
    
    // We need at least 2 candles to compare current vs past
    if (candles.length < 2) return 1.0; 

    // Extract volumes excluding the current candle (we want historical average)
    const historyVolumes = candles.slice(0, candles.length - 1).map(c => c.volume);
    
    // Calculate average of previous volumes
    // If we have 60 candles, take last 20.
    // If we have 5 candles, take all 4 previous.
    const lookback = Math.min(historyVolumes.length, period);
    const relevantHistory = historyVolumes.slice(historyVolumes.length - lookback);
    
    if (relevantHistory.length === 0) return 1.0;

    const sum = relevantHistory.reduce((a, b) => a + b, 0);
    const avgVolume = sum / relevantHistory.length;

    if (avgVolume === 0) return 0;
    return currentVol / avgVolume;
  }

  /**
   * Normalized ATR (NATR)
   * Formula: (ATR(14) / Close) * 100
   * Uses "Dynamic Lookback": If length < 14, averages True Range over available length.
   */
  private static calculateNATR(candles: Candle[]): number {
    const period = 14;
    const close = candles[candles.length - 1]!.close;
    
    if (close === 0) return 0;

    // Calculate True Ranges for all available candles
    // TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))
    const trueRanges: number[] = [];
    for (let i = 1; i < candles.length; i++) {
        const curr = candles[i]!;
        const prev = candles[i-1]!;
        const tr = Math.max(
            curr.high - curr.low,
            Math.abs(curr.high - prev.close),
            Math.abs(curr.low - prev.close)
        );
        trueRanges.push(tr);
    }
    
    if (trueRanges.length === 0) return 0;

    // Use available TRs up to period
    const lookback = Math.min(trueRanges.length, period);
    const recentTRs = trueRanges.slice(trueRanges.length - lookback);
    const avgTR = recentTRs.reduce((a, b) => a + b, 0) / recentTRs.length;

    return (avgTR / close) * 100;
  }

  /**
   * Rolling VWAP Deviation (Z-Score approximation)
   * Formula: (Close - VWAP) / StdDev(Close)
   * Calculates VWAP and StdDev over the entire provided candle window.
   */
  private static calculateVWAPDeviation(candles: Candle[]): number {
     // VWAP = Sum(Volume * Price) / Sum(Volume)
     
     let cumPV = 0;
     let cumVol = 0;
     const closes: number[] = [];

     for (const c of candles) {
         const typicalPrice = (c.high + c.low + c.close) / 3;
         cumPV += typicalPrice * c.volume;
         cumVol += c.volume;
         closes.push(c.close);
     }

     if (cumVol === 0) return 0;
     const vwap = cumPV / cumVol;
     const currentClose = candles[candles.length - 1]!.close;

     // Calculate Standard Deviation of Closes
     const mean = closes.reduce((a, b) => a + b, 0) / closes.length;
     const variance = closes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / closes.length;
     const stdDev = Math.sqrt(variance);

     if (stdDev === 0) return 0;
     
     return (currentClose - vwap) / stdDev;
  }

  /**
   * Stochastic Oscillator (K, D)
   * Standard settings: 14, 3, 3
   * Adapts to shorter arrays by finding High/Low over available history.
   */
  private static calculateStochastic(candles: Candle[], output: 'K'|'D'): number {
      const period = 14;
      
      // Need at least 1 candle
      if (candles.length === 0) return 50;

      // Logic: fast K over period
      // %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
      
      const lookback = Math.min(candles.length, period);
      const relevant = candles.slice(candles.length - lookback);
      
      const lowestLow = Math.min(...relevant.map(c => c.low));
      const highestHigh = Math.max(...relevant.map(c => c.high));
      const currentClose = candles[candles.length - 1]!.close;
      
      let fastK = 50;
      if (highestHigh !== lowestLow) {
          fastK = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
      }
      
      // Constrain to 0-100
      fastK = Math.max(0, Math.min(100, fastK));

      if (output === 'K') return fastK;

      if (output === 'D') {
          // For MVP/Robustness on small data: Returning FastK approx is acceptable
          // as we often don't have enough history for a proper SMA of K.
          return fastK; 
      }
      
      return fastK;
  }

  /**
   * Wick Ratio
   * Upper: (High - Max(Open, Close)) / (High - Low)
   * Lower: (Min(Open, Close) - Low) / (High - Low)
   */
  private static calculateWickRatio(c: Candle, type: 'upper' | 'lower'): number {
      const range = c.high - c.low;
      if (range === 0) return 0;

      const bodyTop = Math.max(c.open, c.close);
      const bodyBottom = Math.min(c.open, c.close);

      if (type === 'upper') {
          return (c.high - bodyTop) / range;
      } else {
          return (bodyBottom - c.low) / range;
      }
  }
}
