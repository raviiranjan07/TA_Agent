import React from 'react';
import { useTradingContext } from '../context/TradingContext';

const Footer = () => {
  const { chartData, theme } = useTradingContext();

  return (
    <div className={`${theme.card} border-t ${theme.border} px-4 py-2`}>
      <div className="flex items-center justify-between text-xs">
        <div className={theme.textSecondary}>
          <span>Timezone: IST (Asia/Kolkata)</span>
          <span className="mx-2">•</span>
          <span>Data updates every minute</span>
          <span className="mx-2">•</span>
          <span>Connected to TimescaleDB</span>
        </div>
        <div className={theme.textSecondary}>
          {chartData.length > 0 && (
            <span>{chartData.length} candles loaded</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default Footer;
