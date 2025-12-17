import { useTradingContext } from '../context/TradingContext';

const StatsBar = () => {
  const { stats, theme, connectionStatus } = useTradingContext();

  if (!stats) return null;

  // Format large numbers
  const formatVolume = (value, decimals = 2) => {
    if (!value) return '0';
    if (value >= 1e9) return `${(value / 1e9).toFixed(decimals)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(decimals)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(decimals)}K`;
    return value.toFixed(decimals);
  };

  // Format price
  const formatPrice = (price) => {
    if (!price) return '$0.00';
    if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    return `$${price.toFixed(2)}`;
  };

  const getConnectionIndicator = () => {
    switch (connectionStatus) {
      case 'connected':
        return <span className="text-green-500 flex items-center gap-1"><span className="animate-pulse">●</span> LIVE</span>;
      case 'error':
        return <span className="text-red-500">● ERROR</span>;
      default:
        return <span className="text-yellow-500 animate-pulse">● CONNECTING...</span>;
    }
  };

  return (
    <div className={`${theme.card} border-b ${theme.border} px-4 py-2`}>
      <div className="grid grid-cols-6 gap-4 text-xs">
        <div>
          <span className={theme.textSecondary}>24h High</span>
          <div className="font-mono font-semibold text-green-500">
            {formatPrice(stats.high_24h)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Low</span>
          <div className="font-mono font-semibold text-red-500">
            {formatPrice(stats.low_24h)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Volume</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            {formatVolume(stats.volume_24h)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>24h Turnover</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            ${formatVolume(stats.quote_volume_24h)}
          </div>
        </div>
        <div>
          <span className={theme.textSecondary}>Trades</span>
          <div className={`${theme.text} font-mono font-semibold`}>
            {stats.trades_24h?.toLocaleString() || '0'}
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
