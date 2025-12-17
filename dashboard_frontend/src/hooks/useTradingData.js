import { useCallback, useEffect, useRef } from 'react';
import { useTradingContext, API_BASE } from '../context/TradingContext';

// Prevent duplicate fetches across re-renders
let isFetching = false;
let lastFetchKey = '';

const useTradingData = () => {
  const {
    selectedPair,
    selectedTimeframe,
    setChartData,
    setStats,
    setIndicators,
    setIsLoading,
    setError,
    autoRefresh,
  } = useTradingContext();

  const intervalRef = useRef(null);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async (force = false) => {
    const fetchKey = `${selectedPair}-${selectedTimeframe}`;

    // Prevent duplicate fetches
    if (isFetching && !force) {
      console.log('Fetch already in progress, skipping...');
      return;
    }

    // Skip if same request was just made
    if (lastFetchKey === fetchKey && !force) {
      const timeSinceLastFetch = Date.now() - (window.lastFetchTime || 0);
      if (timeSinceLastFetch < 1000) {
        console.log('Duplicate fetch prevented');
        return;
      }
    }

    isFetching = true;
    lastFetchKey = fetchKey;
    window.lastFetchTime = Date.now();

    setIsLoading(true);
    setError(null);

    try {
      console.log(`Fetching data for ${selectedPair} ${selectedTimeframe}...`);

      const [candlesRes, statsRes, indicatorsRes] = await Promise.all([
        fetch(`${API_BASE}/api/candles?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`),
        fetch(`${API_BASE}/api/stats?pair=${selectedPair}`),
        fetch(`${API_BASE}/api/indicators?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`)
      ]);

      if (!mountedRef.current) return;

      if (!candlesRes.ok || !statsRes.ok || !indicatorsRes.ok) {
        throw new Error('Failed to fetch trading data');
      }

      const [candlesData, statsData, indicatorsData] = await Promise.all([
        candlesRes.json(),
        statsRes.json(),
        indicatorsRes.json()
      ]);

      if (!mountedRef.current) return;

      // Merge candles with indicators
      const candles = candlesData.candles || [];
      const indicators = indicatorsData.indicators || [];

      const mergedData = candles.map((candle, idx) => {
        const indicator = indicators[idx] || {};
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

      console.log(`Loaded ${mergedData.length} candles`);

      setChartData(mergedData);
      setStats(statsData);
      setIndicators(indicators);
    } catch (error) {
      console.error('Error fetching data:', error);
      if (mountedRef.current) {
        setError(error.message);
      }
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
      isFetching = false;
    }
  }, [selectedPair, selectedTimeframe, setChartData, setStats, setIndicators, setIsLoading, setError]);

  // Initial fetch and when pair/timeframe changes
  useEffect(() => {
    mountedRef.current = true;

    // Small delay to debounce rapid changes
    const timeoutId = setTimeout(() => {
      fetchData(true);
    }, 100);

    return () => {
      clearTimeout(timeoutId);
    };
  }, [selectedPair, selectedTimeframe]);

  // Auto-refresh interval (separate from initial fetch)
  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (autoRefresh) {
      // Set up new interval - every 60 seconds
      intervalRef.current = setInterval(() => {
        fetchData(true);
      }, 60000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoRefresh]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return { fetchData: () => fetchData(true) };
};

export default useTradingData;
