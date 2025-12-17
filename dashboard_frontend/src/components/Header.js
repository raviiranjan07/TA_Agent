import React from 'react';
import { Activity, TrendingUp, TrendingDown, RefreshCw, Download, Moon, Sun, Maximize2 } from 'lucide-react';
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
                <div className={`text-2xl font-bold ${theme.text}`}>
                  ${stats.latest_price?.toFixed(2)}
                </div>
                <div className={`text-xs ${stats.change_24h_percent >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {stats.change_24h_percent >= 0 ? '+' : ''}
                  {stats.change_24h_value?.toFixed(2)} ({stats.change_24h_percent?.toFixed(2)}%)
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
