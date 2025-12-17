import React from 'react';
import { Calendar, Settings, BarChart3 } from 'lucide-react';
import { useTradingContext, TIMEFRAMES } from '../context/TradingContext';

const TimeframeBar = () => {
  const {
    selectedTimeframe,
    setSelectedTimeframe,
    showMA7,
    setShowMA7,
    showMA20,
    setShowMA20,
    showMA50,
    setShowMA50,
    showBB,
    setShowBB,
    showVolume,
    setShowVolume,
    theme,
  } = useTradingContext();

  return (
    <div className={`${theme.card} border-b ${theme.border} px-4 py-2`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Calendar className={`w-4 h-4 ${theme.textSecondary}`} />
          <div className="flex flex-wrap gap-1">
            {TIMEFRAMES.map(tf => (
              <button
                key={tf.value}
                onClick={() => setSelectedTimeframe(tf.value)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-all ${
                  selectedTimeframe === tf.value
                    ? `${theme.active} text-white`
                    : `${theme.hover} ${theme.text}`
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>

        {/* Indicator Toggles */}
        <div className="flex items-center space-x-3">
          <Settings className={`w-4 h-4 ${theme.textSecondary}`} />
          <button
            onClick={() => setShowMA7(!showMA7)}
            className={`text-xs px-2 py-1 rounded transition-all ${
              showMA7 ? 'bg-amber-500 text-white' : `${theme.hover} ${theme.textSecondary}`
            }`}
          >
            MA7
          </button>
          <button
            onClick={() => setShowMA20(!showMA20)}
            className={`text-xs px-2 py-1 rounded transition-all ${
              showMA20 ? 'bg-blue-500 text-white' : `${theme.hover} ${theme.textSecondary}`
            }`}
          >
            MA20
          </button>
          <button
            onClick={() => setShowMA50(!showMA50)}
            className={`text-xs px-2 py-1 rounded transition-all ${
              showMA50 ? 'bg-purple-500 text-white' : `${theme.hover} ${theme.textSecondary}`
            }`}
          >
            MA50
          </button>
          <button
            onClick={() => setShowBB(!showBB)}
            className={`text-xs px-2 py-1 rounded transition-all ${
              showBB ? 'bg-purple-600 text-white' : `${theme.hover} ${theme.textSecondary}`
            }`}
          >
            BB
          </button>
          <button
            onClick={() => setShowVolume(!showVolume)}
            className={`text-xs px-2 py-1 rounded transition-all ${
              showVolume ? 'bg-blue-600 text-white' : `${theme.hover} ${theme.textSecondary}`
            }`}
          >
            <BarChart3 className="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default TimeframeBar;
