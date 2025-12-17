import { useEffect, useRef } from 'react';
import { useTradingContext, WS_URL } from '../context/TradingContext';

const useWebSocket = () => {
  const {
    selectedPair,
    stats,
    setStats,
    autoRefresh,
    setConnectionStatus,
  } = useTradingContext();

  const wsRef = useRef(null);

  useEffect(() => {
    if (!autoRefresh) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'price_update') {
          const pairData = data.prices.find(p => p.pair === selectedPair);

          if (pairData) {
            setStats(prev => prev ? {
              ...prev,
              latest_price: pairData.price,
              latest_time: pairData.time
            } : prev);
          }
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [autoRefresh, selectedPair, setStats, setConnectionStatus]);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN
  };
};

export default useWebSocket;
