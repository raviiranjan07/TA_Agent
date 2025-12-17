import { useCallback, useEffect, useRef } from 'react';
import { useTradingContext, API_BASE } from '../context/TradingContext';

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

  const isFetchingRef = useRef(false);
  const intervalRef = useRef(null);

  const fetchData = useCallback(async () => {
    // Prevent concurrent fetches
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;

    setIsLoading(true);
    setError(null);

    try {
      const [candlesRes, statsRes, indicatorsRes] = await Promise.all([
        fetch(`${API_BASE}/api/candles?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`),
        fetch(`${API_BASE}/api/stats?pair=${selectedPair}`),
        fetch(`${API_BASE}/api/indicators?pair=${selectedPair}&timeframe=${selectedTimeframe}&limit=200`)
      ]);

      if (!candlesRes.ok || !statsRes.ok || !indicatorsRes.ok) {
        throw new Error('Failed to fetch trading data');
      }

      const [candlesData, statsData, indicatorsData] = await Promise.all([
        candlesRes.json(),
        statsRes.json(),
        indicatorsRes.json()
      ]);

      // Merge candles with indicators
      const mergedData = (candlesData.candles || []).map((candle, idx) => {
        const indicator = (indicatorsData.indicators || [])[idx] || {};
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
      setIndicators(indicatorsData.indicators || []);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
      isFetchingRef.current = false;
    }
  }, [selectedPair, selectedTimeframe, setChartData, setStats, setIndicators, setIsLoading, setError]);

  // Initial fetch when pair/timeframe changes
  useEffect(() => {
    fetchData();
  }, [selectedPair, selectedTimeframe]); // Only refetch on pair/timeframe change

  // Auto-refresh interval
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (autoRefresh) {
      intervalRef.current = setInterval(fetchData, 60000);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoRefresh, fetchData]);

  return { fetchData };
};

export default useTradingData;
