import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area, ReferenceLine } from 'recharts';
import { TrendingUp, TrendingDown, Activity, BarChart3, Calendar, RefreshCw, Moon, Sun, Settings, Maximize2, Download, Eye, EyeOff } from 'lucide-react';

const TradingViewDashboard = () => {
  // State management
  const [selectedPair, setSelectedPair] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [chartData, setChartData] = useState([]);
  const [stats, setStats] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [chartType, setChartType] = useState('candlestick');
  
  // Indicator toggles
  const [showMA7, setShowMA7] = useState(true);
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [showVolume, setShowVolume] = useState(true);

  const API_BASE = 'http://localhost:8000';

  // Fetch data from FastAPI
  const fetchData = useCallback(async () => {
    setIsLoading(true);
    try {
      // Fetch candles
      const candlesRes = await fetch(
        `${API_BASE}/api/candles?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`
      );
      const candlesData = await candlesRes.json();
      
      // Fetch stats
      const statsRes = await fetch(`${API_BASE}/api/stats?pair=${selectedPair}`);
      const statsData = await statsRes.json();
      
      // Fetch indicators
      const indicatorsRes = await fetch(
        `${API_BASE}/api/indicators?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`
      );
      const indicatorsData = await indicatorsRes.json();
      
      // Merge data
      const mergedData = candlesData.candles.map((candle, idx) => {
        const indicator = indicatorsData.indicators[idx] || {};
        return {
          ...candle,
          ...indicator,
          displayTime: new Date(candle.time).toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          })
        };
      });
      
      setChartData(mergedData);
      setStats(statsData);
      setIndicators(indicatorsData.indicators);
      
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setIsLoading(false);
    }
  }, [selectedPair, selectedTimeframe]);

  // Auto-refresh
  useEffect(() => {
    fetchData();
    
    if (autoRefresh) {
      const interval = setInterval(fetchData, 60000); // Every minute
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!autoRefresh) return;
    
    const ws = new WebSocket('ws://localhost:8000/ws/live');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'price_update') {
        // Update stats with latest price
        const pairData = data.prices.find(p => p.pair === selectedPair);
        
        if (pairData && stats) {
          setStats(prev => ({
            ...prev,
            latest_price: pairData.price,
            latest_time: pairData.time
          }));
        }
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
    };
    
    return () => {
      ws.close();
    };
  }, [autoRefresh, selectedPair, stats]);

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '3m', label: '3m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '30m', label: '30m' },
    { value: '1h', label: '1h' },
    { value: '2h', label: '2h' },
    { value: '4h', label: '4h' },
    { value: '6h', label: '6h' },
    { value: '12h', label: '12h' },
    { value: '1d', label: '1D' },
    { value: '3d', label: '3D' },
    { value: '1w', label: '1W' }
  ];

  const pairs = ['BTCUSDT', 'ETHUSDT'];

  const theme = {
    bg: isDarkMode ? 'bg-[#131722]' : 'bg-gray-50',
    card: isDarkMode ? 'bg-[#1E222D]' : 'bg-white',
    header: isDarkMode ? 'bg-[#2A2E39]' : 'bg-white',
    text: isDarkMode ? 'text-gray-100' : 'text-gray-900',
    textSecondary: isDarkMode ? 'text-gray-400' : 'text-gray-600',
    border: isDarkMode ? 'border-[#2A2E39]' : 'border-gray-200',
    hover: isDarkMode ? 'hover:bg-[#2A2E39]' : 'hover:bg-gray-100',
    active: isDarkMode ? 'bg-[#2962FF]' : 'bg-blue-500',
  };

  // Custom Candlestick Renderer
  const CandlestickChart = ({ data }) => {
    if (!data || data.length === 0) return null;

    return (
      <ResponsiveContainer width="100%" height={500}>
        <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2962FF" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#2962FF" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#2A2E39' : '#E5E7EB'} />
          
          <XAxis 
            dataKey="displayTime" 
            stroke={isDarkMode ? '#787B86' : '#6B7280'}
            style={{ fontSize: '11px' }}
            interval="preserveStartEnd"
            minTickGap={50}
          />
          
          <YAxis 
            yAxisId="price"
            orientation="right"
            stroke={isDarkMode ? '#787B86' : '#6B7280'}
            style={{ fontSize: '11px' }}
            domain={['dataMin - 100', 'dataMax + 100']}
          />
          
          <YAxis 
            yAxisId="volume"
            orientation="left"
            stroke={isDarkMode ? '#787B86' : '#6B7280'}
            style={{ fontSize: '11px' }}
            domain={[0, 'dataMax * 3']}
            hide
          />
          
          <Tooltip content={<CustomTooltip />} />
          
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

  const CustomTooltip = ({ active, payload }) => {
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

  return (
    <div className={`min-h-screen ${theme.bg} transition-colors duration-300`}>
      {/* TradingView-style Header */}
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
              {pairs.map(pair => (
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
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`p-2 rounded ${theme.hover} transition-colors ${autoRefresh ? 'text-green-500' : theme.textSecondary}`}
              title={autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            
            <button
              onClick={fetchData}
              className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}
              disabled={isLoading}
            >
              <Download className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}
            >
              {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            
            <button className={`p-2 rounded ${theme.hover} transition-colors ${theme.text}`}>
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Timeframe Bar */}
      <div className={`${theme.card} border-b ${theme.border} px-4 py-2`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Calendar className={`w-4 h-4 ${theme.textSecondary}`} />
            <div className="flex flex-wrap gap-1">
              {timeframes.map(tf => (
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

      {/* Stats Bar */}
      {stats && (
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
              <span className={theme.textSecondary}>Updated</span>
              <div className={`${theme.text} font-mono text-green-500 font-semibold`}>
                ● LIVE (IST)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Chart Area */}
      <div className={`${theme.card} p-4`}>
        {isLoading && chartData.length === 0 ? (
          <div className="flex items-center justify-center h-[500px]">
            <div className={`${theme.textSecondary} text-center`}>
              <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-2" />
              <p>Loading chart data...</p>
            </div>
          </div>
        ) : chartData.length === 0 ? (
          <div className="flex items-center justify-center h-[500px]">
            <div className={`${theme.textSecondary} text-center`}>
              <p>No data available</p>
              <p className="text-xs mt-2">Make sure your pipeline is running</p>
            </div>
          </div>
        ) : (
          <CandlestickChart data={chartData} />
        )}
      </div>

      {/* Footer Info */}
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
    </div>
  );
};

export default TradingViewDashboard;