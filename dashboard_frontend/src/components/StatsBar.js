import React from 'react';
import { useTradingContext } from '../context/TradingContext';

const StatsBar = () => {
  const { stats, theme, connectionStatus } = useTradingContext();

  if (!stats) return null;

  const getConnectionIndicator = () => {
    switch (connectionStatus) {
      case 'connected':
        return <span className="text-green-500">● LIVE (IST)</span>;
      case 'error':
        return <span className="text-red-500">● ERROR</span>;
      default:
        return <span className="text-yellow-500">● CONNECTING...</span>;
    }
  };

  return (
    <div className={`${theme.card} border-b ${theme.border} px-4 py-2`}>
      <div className="grid grid-cols-6 gap-4 text-xs">
        <div>
          <span className={theme.textSecondary}>24h High</span>
          <div className={`${theme.text} font-mono font-semibold text-green-500`}>
            ${stats.high_24h?.toFixed(2)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Low</span>
          <div className={`${theme.text} font-mono font-semibold text-red-500`}>
            ${stats.low_24h?.toFixed(2)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Volume (Base)</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            {(stats.volume_24h / 1000).toFixed(2)}K
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Volume (Quote)</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            ${(stats.quote_volume_24h / 1000000).toFixed(2)}M
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>Trades</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            {stats.trades_24h?.toLocaleString()}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>Status</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            {getConnectionIndicator()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsBar;
