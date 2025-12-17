import { useCallback, useEffect } from 'react';
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

  const fetchData = useCallback(async () => {
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
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  }, [selectedPair, selectedTimeframe, setChartData, setStats, setIndicators, setIsLoading, setError]);

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchData();

    if (autoRefresh) {
      const interval = setInterval(fetchData, 60000);
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  return { fetchData };
};

export default useTradingData;
