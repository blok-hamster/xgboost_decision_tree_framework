
import fs from 'fs';
import { stringify } from 'csv-stringify/sync';
import { DataLoader } from './data_loader';
import { FeatureExtractor } from './features';
import { CONFIG } from './config';

async function runPipeline() {
  console.log("🚀 Starting Advanced KOL Classification Pipeline...");
  

  // 1. Validate Inputs
  if (!fs.existsSync(CONFIG.ANALYSIS_DIR) || !fs.existsSync(CONFIG.OHLCV_DIR)) {
      console.error("❌ Valid input directories not found!");
      console.error(`Status: Analysis Dir: ${fs.existsSync(CONFIG.ANALYSIS_DIR)}, OHLCV Dir: ${fs.existsSync(CONFIG.OHLCV_DIR)}`);
      return;
  }
  
  // 2. Load Trades
  let trades;
  try {
      trades = DataLoader.loadTrades();
      console.log(`✅ Loaded ${trades.length} labeled trades.`);
  } catch (error) {
      console.error("❌ Failed to load trades:", error);
      return;
  }

  const results: any[] = [];
  let skippedCount = 0;
  let successCount = 0;

  // 2. Process each trade
  for (const trade of trades) {
      const candles = DataLoader.loadCandlesForTrade(trade.token_address, trade.buy_timestamp);
      
      if (!candles) {
          // console.log(`⚠️ No OHLCV data for ${trade.token_address} before timestamp.`);
          skippedCount++;
          continue;
      }

      // We need at least 1 candle to do anything
      if (candles.length === 0) {
          skippedCount++;
          continue;
      }

      try {
          const features = FeatureExtractor.extract(candles);
          
          results.push({
             token_address: trade.token_address,
             buy_timestamp: trade.buy_timestamp,
             // Features
             rvol: features.rvol.toFixed(4),
             natr: features.natr.toFixed(4),
             vwap_dev: features.vwap_deviation.toFixed(4),
             stoch_k: features.k_percentage.toFixed(4),
             stoch_d: features.d_percentage.toFixed(4),
             upper_wick: features.upper_wick_ratio.toFixed(4),
             lower_wick: features.lower_wick_ratio.toFixed(4),
             market_cap: features.market_cap.toFixed(2),
             token_age_hours: features.visible_token_age.toFixed(2),
             // Metadata
             data_points: candles.length,
             label: trade.performance_label
          });
          
          successCount++;
          
      } catch (err) {
          console.error(`❌ Error processing ${trade.token_address}:`, err);
          skippedCount++;
      }
  }

  console.log(`\n📊 Processing Complete:`);
  console.log(`   - Success: ${successCount}`);
  console.log(`   - Skipped: ${skippedCount} (Missing data or 0 length)`);

  if (results.length === 0) {
      console.error("❌ No data generated. Exiting.");
      return;
  }

  // 3. Write to CSV
  const csvOutput = stringify(results, { header: true });
  fs.writeFileSync(CONFIG.OUTPUT_FILE, csvOutput);
  
  console.log(`\n✅ Saved training data to: ${CONFIG.OUTPUT_FILE}`);
}

runPipeline().catch(console.error);
