import { useEffect, useRef } from 'react';
import { useTradingContext, WS_URL } from '../context/TradingContext';

const useWebSocket = () => {
  const {
    selectedPair,
    setStats,
    autoRefresh,
    setConnectionStatus,
  } = useTradingContext();

  const wsRef = useRef(null);
  const selectedPairRef = useRef(selectedPair);

  // Keep ref updated
  useEffect(() => {
    selectedPairRef.current = selectedPair;
  }, [selectedPair]);

  useEffect(() => {
    if (!autoRefresh) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setConnectionStatus('disconnected');
      }
      return;
    }

    // Don't reconnect if already connected
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    let reconnectTimeout = null;
    let isCleaningUp = false;

    const connect = () => {
      if (isCleaningUp) return;

      try {
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
              const pairData = data.prices.find(p => p.pair === selectedPairRef.current);

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
          wsRef.current = null;

          // Reconnect after 5 seconds if not cleaning up
          if (!isCleaningUp && autoRefresh) {
            reconnectTimeout = setTimeout(connect, 5000);
          }
        };
      } catch (error) {
        console.error('WebSocket connection error:', error);
        setConnectionStatus('error');
      }
    };

    connect();

    return () => {
      isCleaningUp = true;
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [autoRefresh, setConnectionStatus, setStats]);

  return {
    isConnected: wsRef.current?.readyState === WebSocket.OPEN
  };
};

export default useWebSocket;
