import React from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
} from 'recharts';
import { RefreshCw } from 'lucide-react';
import { useTradingContext } from '../context/TradingContext';

const CustomTooltip = ({ active, payload, theme }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isGreen = data.close >= data.open;

    return (
      <div className={`${theme.card} p-4 rounded-lg shadow-xl border ${theme.border}`}>
        <p className={`${theme.text} text-xs font-semibold mb-2`}>{data.displayTime} IST</p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div>
            <span className={theme.textSecondary}>Open:</span>
            <span className={`${theme.text} font-mono ml-2`}>${data.open?.toFixed(2)}</span>
          </div>
          <div>
            <span className={theme.textSecondary}>High:</span>
            <span className="text-green-500 font-mono ml-2">${data.high?.toFixed(2)}</span>
          </div>
          <div>
            <span className={theme.textSecondary}>Low:</span>
            <span className="text-red-500 font-mono ml-2">${data.low?.toFixed(2)}</span>
          </div>
          <div>
            <span className={theme.textSecondary}>Close:</span>
            <span className={`${isGreen ? 'text-green-500' : 'text-red-500'} font-mono ml-2`}>
              ${data.close?.toFixed(2)}
            </span>
          </div>
          {data.volume && (
            <div className="col-span-2">
              <span className={theme.textSecondary}>Volume:</span>
              <span className={`${theme.text} font-mono ml-2`}>{data.volume?.toFixed(2)}</span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

const ChartContent = ({ data }) => {
  const {
    stats,
    showMA7,
    showMA20,
    showMA50,
    showBB,
    showVolume,
    theme,
  } = useTradingContext();

  if (!data || data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={500}>
      <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2962FF" stopOpacity={0.8} />
            <stop offset="95%" stopColor="#2962FF" stopOpacity={0.1} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke={theme.gridStroke} />

        <XAxis
          dataKey="displayTime"
          stroke={theme.axisStroke}
          style={{ fontSize: '11px' }}
          interval="preserveStartEnd"
          minTickGap={50}
        />

        <YAxis
          yAxisId="price"
          orientation="right"
          stroke={theme.axisStroke}
          style={{ fontSize: '11px' }}
          domain={['dataMin - 100', 'dataMax + 100']}
        />

        <YAxis
          yAxisId="volume"
          orientation="left"
          stroke={theme.axisStroke}
          style={{ fontSize: '11px' }}
          domain={[0, 'dataMax * 3']}
          hide
        />

        <Tooltip content={<CustomTooltip theme={theme} />} />

        {/* Volume bars */}
        {showVolume && (
          <Bar
            yAxisId="volume"
            dataKey="volume"
            fill="url(#volumeGradient)"
            opacity={0.3}
          />
        )}

        {/* Bollinger Bands */}
        {showBB && (
          <>
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="bb_upper"
              stroke="#9333EA"
              fill="#9333EA"
              fillOpacity={0.1}
              strokeWidth={1}
              dot={false}
            />
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="bb_lower"
              stroke="#9333EA"
              fill="#9333EA"
              fillOpacity={0.1}
              strokeWidth={1}
              dot={false}
            />
          </>
        )}

        {/* Moving Averages */}
        {showMA7 && (
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="ma7"
            stroke="#F59E0B"
            strokeWidth={2}
            dot={false}
            name="MA7"
          />
        )}

        {showMA20 && (
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="ma20"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            name="MA20"
          />
        )}

        {showMA50 && (
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="ma50"
            stroke="#8B5CF6"
            strokeWidth={2}
            dot={false}
            name="MA50"
          />
        )}

        {/* Price line */}
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          stroke={stats?.change_24h_percent >= 0 ? '#10B981' : '#EF4444'}
          strokeWidth={2}
          dot={false}
          name="Close"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

const CandlestickChart = () => {
  const { chartData, isLoading, theme } = useTradingContext();

  if (isLoading && chartData.length === 0) {
    return (
      <div className={`${theme.card} p-4`}>
        <div className="flex items-center justify-center h-[500px]">
          <div className={`${theme.textSecondary} text-center`}>
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
            <p>Loading chart data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className={`${theme.card} p-4`}>
        <div className="flex items-center justify-center h-[500px]">
          <div className={`${theme.textSecondary} text-center`}>
            <p>No data available</p>
            <p className="text-xs mt-2">Make sure your pipeline is running</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`${theme.card} p-4`}>
      <ChartContent data={chartData} />
    </div>
  );
};

export default CandlestickChart;
