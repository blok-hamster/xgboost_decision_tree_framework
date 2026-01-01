
export interface TradeLabel {
  token_address: string;
  buy_timestamp: number; // Milliseconds
  performance_label: 'good' | 'bad' | 'neutral';
}

export interface Candle {
  timestamp: number; // Seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  market_cap?: number; // Estimated
}

export interface AdvancedFeatures {
  rvol: number;
  k_percentage: number;
  d_percentage: number;
  natr: number;
  vwap_deviation: number;
  upper_wick_ratio: number;
  lower_wick_ratio: number;
  market_cap: number;
  visible_token_age: number; // Hours of data available before buy
}
