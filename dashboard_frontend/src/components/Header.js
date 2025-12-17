import { Activity, TrendingUp, TrendingDown, RefreshCw, Download, Moon, Sun, Maximize2, Clock } from 'lucide-react';
import { useTradingContext, TRADING_PAIRS } from '../context/TradingContext';

const Header = ({ onRefresh, isLoading }) => {
  const {
    selectedPair,
    setSelectedPair,
    stats,
    autoRefresh,
    toggleAutoRefresh,
    isDarkMode,
    toggleTheme,
    theme,
  } = useTradingContext();

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  // Format price with proper decimals based on value
  const formatPrice = (price) => {
    if (!price) return '$0.00';
    if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(6)}`;
  };

  // Format last update time
  const formatLastUpdate = (timeStr) => {
    if (!timeStr) return '';
    try {
      const date = new Date(timeStr);
      return date.toLocaleString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    } catch {
      return '';
    }
  };

  return (
    <div className={`${theme.header} border-b ${theme.border} px-4 py-3`}>
      <div className="flex items-center justify-between">
        {/* Logo & Pair */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-6 h-6 text-blue-500" />
            <span className={`text-lg font-bold ${theme.text}`}>TradingView Pro</span>
          </div>

          {/* Pair Selector */}
          <div className="flex gap-1">
            {TRADING_PAIRS.map(pair => (
              <button
                key={pair}
                onClick={() => setSelectedPair(pair)}
                className={`px-3 py-1.5 rounded font-semibold text-sm transition-all ${
                  selectedPair === pair
                    ? `${theme.active} text-white`
                    : `${theme.hover} ${theme.text}`
                }`}
              >
                {pair.replace('USDT', '/USDT')}
              </button>
            ))}
          </div>

          {/* Price Display */}
          {stats && (
            <div className="flex items-center space-x-3 ml-4">
              <div>
                <div className={`text-2xl font-bold font-mono ${theme.text}`}>
                  {formatPrice(stats.latest_price)}
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-mono ${stats.change_24h_percent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {stats.change_24h_percent >= 0 ? '+' : ''}
                    {stats.change_24h_value?.toFixed(2)} ({stats.change_24h_percent?.toFixed(2)}%)
                  </span>
                  {stats.latest_time && (
                    <span className={`text-xs ${theme.textSecondary} flex items-center gap-1`}>
                      <Clock className="w-3 h-3" />
                      {formatLastUpdate(stats.latest_time)}
                    </span>
                  )}
                </div>
              </div>
              {stats.change_24h_percent >= 0 ? (
                <TrendingUp className="w-5 h-5 text-green-500" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-500" />
              )}
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={toggleAutoRefresh}
            className={`p-2 rounded ${theme.hover} transition-colors ${autoRefresh ? 'text-green-500' : theme.textSecondary}`}
            title={autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>

          <button
            onClick={onRefresh}
            className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}
            disabled={isLoading}
            title="Download/Refresh data"
          >
            <Download className="w-4 h-4" />
          </button>

          <button
            onClick={toggleTheme}
            className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}
            title={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          <button
            onClick={handleFullscreen}
            className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}
            title="Toggle fullscreen"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Header;
