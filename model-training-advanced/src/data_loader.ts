
import fs from 'fs';
import path from 'path';
import { parse } from 'csv-parse/sync';
import { CONFIG } from './config';
import { TradeLabel, Candle } from './types';

export class DataLoader {
  
  /**
   * Loads all labeled trades from the latest analysis JSON file.
   */
  static loadTrades(): TradeLabel[] {
    const files = fs.readdirSync(CONFIG.ANALYSIS_DIR)
      .filter(f => f.startsWith('token-positions-') && f.endsWith('.json'))
      .sort()
      .reverse(); // Get latest first

    if (files.length === 0) {
      throw new Error('No analysis files found in ' + CONFIG.ANALYSIS_DIR);
    }

    const targetFile = path.join(CONFIG.ANALYSIS_DIR, files[0]);
    console.log(`Loading trades from: ${targetFile}`);
    
    const rawData = fs.readFileSync(targetFile, 'utf-8');
    const trades = JSON.parse(rawData);
    
    // Filter down to just what we need
    return trades.map((t: any) => ({
      token_address: t.token_address,
      buy_timestamp: t.buy_timestamp,
      performance_label: t.performance_label
    }));
  }

  private static fileCache: Map<string, string> | null = null;

  /**
   * Build a cache mapping TokenAddress -> Filename to avoid O(N*M) directory scans.
   */
  private static buildFileCache() {
      console.log("📂 Building OHLCV file cache...");
      const files = fs.readdirSync(CONFIG.OHLCV_DIR);
      DataLoader.fileCache = new Map();
      
      for (const f of files) {
          if (!f.endsWith('.csv')) continue;
          // Filename format: {tokenAddress}_{timestamp}_...
          // We assume the token address is the first part before the first underscore.
          // However, some addresses might have underscores? unlikely for Solana.
          // Safer: standard solana addresses are base58, no underscores.
          const tokenAddress = f.split('_')[0];
          DataLoader.fileCache.set(tokenAddress, f);
      }
      console.log(`✅ Cached ${DataLoader.fileCache.size} OHLCV files.`);
  }

  /**
   * Finds and loads the OHLCV candles for a specific trade.
   */
  static loadCandlesForTrade(tokenAddress: string, buyTimestampMs: number): Candle[] | null {
    if (!DataLoader.fileCache) {
        DataLoader.buildFileCache();
    }

    const validFile = DataLoader.fileCache!.get(tokenAddress);

    if (!validFile) {
      return null;
    }

    const filePath = path.join(CONFIG.OHLCV_DIR, validFile);
    const fileContent = fs.readFileSync(filePath, 'utf-8');

    // 2. Parse CSV
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      cast: true // Auto-convert numbers
    });

    const buyTimeSec = Math.floor(buyTimestampMs / 1000);

    // 3. Filter: Get candles strictly BEFORE the buy timestamp
    // We want the view the model would have had *at the moment of decision*.
    const historicalCandles = records.filter((r: any) => r.timestamp < buyTimeSec);

    // 4. Sort by timestamp ascending just in case
    historicalCandles.sort((a: any, b: any) => a.timestamp - b.timestamp);

    // 5. Return the last 60 candles (1 hour window) or whatever is avail
    // If we have 100 candles, we take the last 60.
    // If we have 5 candles, we take 5.
    const sliceStart = Math.max(0, historicalCandles.length - 60);
    const last60 = historicalCandles.slice(sliceStart);

    if (last60.length === 0) {
        return null;
    }

    return last60.map((r: any) => ({
      timestamp: Number(r.timestamp),
      open: Number(r.open),
      high: Number(r.high),
      low: Number(r.low),
      close: Number(r.close),
      volume: Number(r.volume),
      market_cap: Number(r.close) * 1_000_000_000 // Standard 1B Supply Assumption
    }));
  }
}
