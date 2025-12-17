import React, { createContext, useContext, useState, useCallback } from 'react';

const TradingContext = createContext(null);

export const TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT'];

export const TIMEFRAMES = [
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

export const API_BASE = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000/ws/live';

export const TradingProvider = ({ children }) => {
  // Core trading state
  const [selectedPair, setSelectedPair] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [chartData, setChartData] = useState([]);
  const [stats, setStats] = useState(null);
  const [indicators, setIndicators] = useState([]);

  // UI state
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);

  // Indicator toggles
  const [showMA7, setShowMA7] = useState(true);
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [showVolume, setShowVolume] = useState(true);

  // Theme configuration
  const theme = {
    bg: isDarkMode ? 'bg-[#131722]' : 'bg-gray-50',
    card: isDarkMode ? 'bg-[#1E222D]' : 'bg-white',
    header: isDarkMode ? 'bg-[#2A2E39]' : 'bg-white',
    text: isDarkMode ? 'text-gray-100' : 'text-gray-900',
    textSecondary: isDarkMode ? 'text-gray-400' : 'text-gray-600',
    border: isDarkMode ? 'border-[#2A2E39]' : 'border-gray-200',
    hover: isDarkMode ? 'hover:bg-[#2A2E39]' : 'hover:bg-gray-100',
    active: isDarkMode ? 'bg-[#2962FF]' : 'bg-blue-500',
    gridStroke: isDarkMode ? '#2A2E39' : '#E5E7EB',
    axisStroke: isDarkMode ? '#787B86' : '#6B7280',
  };

  const toggleTheme = useCallback(() => {
    setIsDarkMode(prev => !prev);
  }, []);

  const toggleAutoRefresh = useCallback(() => {
    setAutoRefresh(prev => !prev);
  }, []);

  const value = {
    // Trading state
    selectedPair,
    setSelectedPair,
    selectedTimeframe,
    setSelectedTimeframe,
    chartData,
    setChartData,
    stats,
    setStats,
    indicators,
    setIndicators,

    // UI state
    isDarkMode,
    isLoading,
    setIsLoading,
    autoRefresh,
    connectionStatus,
    setConnectionStatus,
    error,
    setError,
    theme,

    // Indicator toggles
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

    // Actions
    toggleTheme,
    toggleAutoRefresh,
  };

  return (
    <TradingContext.Provider value={value}>
      {children}
    </TradingContext.Provider>
  );
};

export const useTradingContext = () => {
  const context = useContext(TradingContext);
  if (!context) {
    throw new Error('useTradingContext must be used within a TradingProvider');
  }
  return context;
};

export default TradingContext;
