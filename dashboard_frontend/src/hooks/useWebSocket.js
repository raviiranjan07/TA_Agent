import { useEffect, useRef } from 'react';
import { useTradingContext, WS_URL } from '../context/TradingContext';

// Singleton WebSocket connection to prevent multiple connections
let globalWs = null;
let globalWsListeners = new Set();

const useWebSocket = () => {
  const {
    selectedPair,
    setStats,
    autoRefresh,
    setConnectionStatus,
  } = useTradingContext();

  const selectedPairRef = useRef(selectedPair);
  const setStatsRef = useRef(setStats);
  const setConnectionStatusRef = useRef(setConnectionStatus);

  // Keep refs updated
  useEffect(() => {
    selectedPairRef.current = selectedPair;
  }, [selectedPair]);

  useEffect(() => {
    setStatsRef.current = setStats;
    setConnectionStatusRef.current = setConnectionStatus;
  }, [setStats, setConnectionStatus]);

  useEffect(() => {
    if (!autoRefresh) {
      // Close global connection when autoRefresh is disabled
      if (globalWs) {
        globalWs.close();
        globalWs = null;
      }
      setConnectionStatusRef.current('disconnected');
      return;
    }

    const handleMessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'price_update') {
          const pairData = data.prices.find(p => p.pair === selectedPairRef.current);
          if (pairData) {
            setStatsRef.current(prev => prev ? {
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

    // Add this component's listener
    globalWsListeners.add(handleMessage);

    // Create connection if doesn't exist
    if (!globalWs || globalWs.readyState === WebSocket.CLOSED) {
      try {
        globalWs = new WebSocket(WS_URL);

        globalWs.onopen = () => {
          console.log('WebSocket connected (singleton)');
          setConnectionStatusRef.current('connected');
        };

        globalWs.onmessage = (event) => {
          // Broadcast to all listeners
          globalWsListeners.forEach(listener => listener(event));
        };

        globalWs.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnectionStatusRef.current('error');
        };

        globalWs.onclose = () => {
          console.log('WebSocket disconnected');
          setConnectionStatusRef.current('disconnected');
          globalWs = null;
        };
      } catch (error) {
        console.error('WebSocket connection error:', error);
        setConnectionStatusRef.current('error');
      }
    } else if (globalWs.readyState === WebSocket.OPEN) {
      setConnectionStatusRef.current('connected');
    }

    // Cleanup - remove listener but don't close shared connection
    return () => {
      globalWsListeners.delete(handleMessage);
      // Only close if no more listeners
      if (globalWsListeners.size === 0 && globalWs) {
        globalWs.close();
        globalWs = null;
      }
    };
  }, [autoRefresh]);

  return {
    isConnected: globalWs?.readyState === WebSocket.OPEN
  };
};

export default useWebSocket;
