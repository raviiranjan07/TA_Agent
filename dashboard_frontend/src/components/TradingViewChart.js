import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { useTradingContext } from '../context/TradingContext';

const TradingViewChart = () => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef({});

  // State for crosshair OHLCV display
  const [ohlcData, setOhlcData] = useState(null);

  const {
    chartData,
    indicators,
    isDarkMode,
    showMA7,
    showMA20,
    showMA50,
    showBB,
    showVolume,
    stats,
    selectedPair,
    selectedTimeframe,
  } = useTradingContext();

  // Format price with proper decimals
  const formatPrice = useCallback((price, pair) => {
    if (price == null || isNaN(price)) return '—';
    const p = parseFloat(price);
    // BTC typically needs 2 decimals, other coins may need more
    if (pair && pair.includes('BTC')) {
      return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
  }, []);

  // Format volume
  const formatVolume = useCallback((vol) => {
    if (vol == null || isNaN(vol)) return '—';
    const num = parseFloat(vol);
    if (num >= 1e9) return (num / 1e9).toFixed(3) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(3) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(3) + 'K';
    return num.toFixed(4);
  }, []);

  // Parse timestamp correctly - API returns ISO string with timezone
  const parseTimestamp = useCallback((candle) => {
    // The API returns 'time' as ISO string like "2024-12-21T10:30:00+05:30"
    // and 'timestamp' as milliseconds

    if (candle.timestamp && typeof candle.timestamp === 'number') {
      // Convert milliseconds to seconds for lightweight-charts
      return Math.floor(candle.timestamp / 1000);
    }

    if (candle.time) {
      // Parse ISO string
      const date = new Date(candle.time);
      if (!isNaN(date.getTime())) {
        return Math.floor(date.getTime() / 1000);
      }
    }

    return null;
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Cleanup existing chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      seriesRef.current = {};
    }

    const container = chartContainerRef.current;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: 550,
      layout: {
        background: { type: 'solid', color: '#131722' },
        textColor: '#d1d4dc',
        fontSize: 12,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif",
      },
      grid: {
        vertLines: {
          color: 'rgba(42, 46, 57, 0.6)',
          style: 1,
          visible: true,
        },
        horzLines: {
          color: 'rgba(42, 46, 57, 0.6)',
          style: 1,
          visible: true,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          width: 1,
          color: '#505050',
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: '#2962FF',
        },
        horzLine: {
          width: 1,
          color: '#505050',
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: '#2962FF',
        },
      },
      rightPriceScale: {
        borderColor: '#2A2E39',
        borderVisible: true,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
        autoScale: true,
        mode: 0, // Normal mode
      },
      timeScale: {
        borderColor: '#2A2E39',
        borderVisible: true,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 10,
        barSpacing: 10,
        minBarSpacing: 5,
        fixLeftEdge: false,
        fixRightEdge: false,
        tickMarkFormatter: (time) => {
          const date = new Date(time * 1000);
          const hours = date.getHours().toString().padStart(2, '0');
          const minutes = date.getMinutes().toString().padStart(2, '0');
          const day = date.getDate();
          const month = date.toLocaleString('en', { month: 'short' });

          // Show date for daily, otherwise show time
          if (selectedTimeframe === '1d' || selectedTimeframe === '3d' || selectedTimeframe === '1w') {
            return `${day} ${month}`;
          }
          return `${hours}:${minutes}`;
        },
      },
      localization: {
        timeFormatter: (time) => {
          const date = new Date(time * 1000);
          return date.toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
          });
        },
        priceFormatter: (price) => {
          if (selectedPair && selectedPair.includes('BTC')) {
            return price.toFixed(2);
          }
          return price.toFixed(4);
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    chartRef.current = chart;

    // === CANDLESTICK SERIES - TradingView Colors ===
    seriesRef.current.candlestick = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      priceLineVisible: true,
      lastValueVisible: true,
      priceFormat: {
        type: 'price',
        precision: selectedPair && selectedPair.includes('BTC') ? 2 : 4,
        minMove: selectedPair && selectedPair.includes('BTC') ? 0.01 : 0.0001,
      },
    });

    // === VOLUME SERIES ===
    seriesRef.current.volume = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
      visible: false,
    });

    // === MOVING AVERAGES ===
    seriesRef.current.ma7 = chart.addLineSeries({
      color: '#F7931A',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    seriesRef.current.ma20 = chart.addLineSeries({
      color: '#2962FF',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    seriesRef.current.ma50 = chart.addLineSeries({
      color: '#7B1FA2',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 3,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    // === BOLLINGER BANDS ===
    seriesRef.current.bbUpper = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 0,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    seriesRef.current.bbMiddle = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    seriesRef.current.bbLower = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 0,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    // === CROSSHAIR MOVE HANDLER ===
    chart.subscribeCrosshairMove((param) => {
      if (!param || !param.time || !param.seriesData) {
        setOhlcData(null);
        return;
      }

      const candleData = param.seriesData.get(seriesRef.current.candlestick);
      const volumeData = param.seriesData.get(seriesRef.current.volume);

      if (candleData) {
        setOhlcData({
          time: param.time,
          open: candleData.open,
          high: candleData.high,
          low: candleData.low,
          close: candleData.close,
          volume: volumeData?.value || 0,
          change: candleData.close - candleData.open,
          changePercent: ((candleData.close - candleData.open) / candleData.open) * 100,
        });
      }
    });

    // === RESIZE HANDLER ===
    const handleResize = () => {
      if (container && chart) {
        chart.applyOptions({ width: container.clientWidth });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        seriesRef.current = {};
      }
    };
  }, [selectedPair, selectedTimeframe]);

  // Update chart data when chartData changes
  useEffect(() => {
    if (!chartData || chartData.length === 0 || !seriesRef.current.candlestick) return;

    // Process candle data
    const processedCandles = [];
    const processedVolumes = [];
    const seenTimes = new Set();

    for (const candle of chartData) {
      const time = parseTimestamp(candle);
      if (time === null || seenTimes.has(time)) continue;
      seenTimes.add(time);

      const open = parseFloat(candle.open);
      const high = parseFloat(candle.high);
      const low = parseFloat(candle.low);
      const close = parseFloat(candle.close);
      const volume = parseFloat(candle.volume) || 0;

      if (isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) continue;

      processedCandles.push({ time, open, high, low, close });
      processedVolumes.push({
        time,
        value: volume,
        color: close >= open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      });
    }

    // Sort by time
    processedCandles.sort((a, b) => a.time - b.time);
    processedVolumes.sort((a, b) => a.time - b.time);

    // Set data
    seriesRef.current.candlestick.setData(processedCandles);

    // Volume
    if (showVolume) {
      seriesRef.current.volume.setData(processedVolumes);
      seriesRef.current.volume.applyOptions({ visible: true });
    } else {
      seriesRef.current.volume.setData([]);
      seriesRef.current.volume.applyOptions({ visible: false });
    }

    // === INDICATORS ===
    const indicatorData = indicators && indicators.length > 0 ? indicators : chartData;

    // MA7
    if (seriesRef.current.ma7) {
      if (showMA7) {
        const ma7Data = [];
        const seenMA7 = new Set();
        for (const d of indicatorData) {
          if (d.ma7 == null) continue;
          const time = parseTimestamp(d);
          if (time === null || seenMA7.has(time)) continue;
          seenMA7.add(time);
          const value = parseFloat(d.ma7);
          if (!isNaN(value)) ma7Data.push({ time, value });
        }
        ma7Data.sort((a, b) => a.time - b.time);
        seriesRef.current.ma7.setData(ma7Data);
        seriesRef.current.ma7.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma7.setData([]);
        seriesRef.current.ma7.applyOptions({ visible: false });
      }
    }

    // MA20
    if (seriesRef.current.ma20) {
      if (showMA20) {
        const ma20Data = [];
        const seenMA20 = new Set();
        for (const d of indicatorData) {
          if (d.ma20 == null) continue;
          const time = parseTimestamp(d);
          if (time === null || seenMA20.has(time)) continue;
          seenMA20.add(time);
          const value = parseFloat(d.ma20);
          if (!isNaN(value)) ma20Data.push({ time, value });
        }
        ma20Data.sort((a, b) => a.time - b.time);
        seriesRef.current.ma20.setData(ma20Data);
        seriesRef.current.ma20.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma20.setData([]);
        seriesRef.current.ma20.applyOptions({ visible: false });
      }
    }

    // MA50
    if (seriesRef.current.ma50) {
      if (showMA50) {
        const ma50Data = [];
        const seenMA50 = new Set();
        for (const d of indicatorData) {
          if (d.ma50 == null) continue;
          const time = parseTimestamp(d);
          if (time === null || seenMA50.has(time)) continue;
          seenMA50.add(time);
          const value = parseFloat(d.ma50);
          if (!isNaN(value)) ma50Data.push({ time, value });
        }
        ma50Data.sort((a, b) => a.time - b.time);
        seriesRef.current.ma50.setData(ma50Data);
        seriesRef.current.ma50.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma50.setData([]);
        seriesRef.current.ma50.applyOptions({ visible: false });
      }
    }

    // Bollinger Bands
    if (seriesRef.current.bbUpper && seriesRef.current.bbMiddle && seriesRef.current.bbLower) {
      if (showBB) {
        const bbUpperData = [];
        const bbMiddleData = [];
        const bbLowerData = [];
        const seenBB = new Set();

        for (const d of indicatorData) {
          const time = parseTimestamp(d);
          if (time === null || seenBB.has(time)) continue;
          seenBB.add(time);

          if (d.bb_upper != null) {
            const val = parseFloat(d.bb_upper);
            if (!isNaN(val)) bbUpperData.push({ time, value: val });
          }
          if (d.bb_middle != null) {
            const val = parseFloat(d.bb_middle);
            if (!isNaN(val)) bbMiddleData.push({ time, value: val });
          }
          if (d.bb_lower != null) {
            const val = parseFloat(d.bb_lower);
            if (!isNaN(val)) bbLowerData.push({ time, value: val });
          }
        }

        bbUpperData.sort((a, b) => a.time - b.time);
        bbMiddleData.sort((a, b) => a.time - b.time);
        bbLowerData.sort((a, b) => a.time - b.time);

        seriesRef.current.bbUpper.setData(bbUpperData);
        seriesRef.current.bbMiddle.setData(bbMiddleData);
        seriesRef.current.bbLower.setData(bbLowerData);
        seriesRef.current.bbUpper.applyOptions({ visible: true });
        seriesRef.current.bbMiddle.applyOptions({ visible: true });
        seriesRef.current.bbLower.applyOptions({ visible: true });
      } else {
        seriesRef.current.bbUpper.setData([]);
        seriesRef.current.bbMiddle.setData([]);
        seriesRef.current.bbLower.setData([]);
        seriesRef.current.bbUpper.applyOptions({ visible: false });
        seriesRef.current.bbMiddle.applyOptions({ visible: false });
        seriesRef.current.bbLower.applyOptions({ visible: false });
      }
    }

    // Fit content to view
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [chartData, indicators, showMA7, showMA20, showMA50, showBB, showVolume, parseTimestamp]);

  // Get display data (crosshair or latest)
  const latestCandle = chartData && chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const displayData = ohlcData || (latestCandle ? {
    open: parseFloat(latestCandle.open),
    high: parseFloat(latestCandle.high),
    low: parseFloat(latestCandle.low),
    close: parseFloat(latestCandle.close),
    volume: parseFloat(latestCandle.volume) || 0,
    change: parseFloat(latestCandle.close) - parseFloat(latestCandle.open),
    changePercent: ((parseFloat(latestCandle.close) - parseFloat(latestCandle.open)) / parseFloat(latestCandle.open)) * 100,
  } : null);

  const isPositive = displayData ? displayData.close >= displayData.open : true;

  return (
    <div className="relative w-full" style={{ backgroundColor: '#131722' }}>
      {/* OHLCV Header - TradingView Style */}
      <div
        className="absolute top-1 left-2 z-20 flex flex-wrap items-center gap-x-3 text-xs"
        style={{ fontFamily: "'Trebuchet MS', sans-serif" }}
      >
        {/* Symbol */}
        <div className="flex items-center gap-1.5">
          <span className="text-white font-semibold text-sm">{selectedPair}</span>
          <span className="text-gray-400 text-[10px] px-1 py-0.5 rounded bg-gray-800">
            {selectedTimeframe.toUpperCase()}
          </span>
        </div>

        {/* OHLCV Values */}
        {displayData && (
          <>
            <div className="flex items-center gap-0.5">
              <span className="text-gray-500">O</span>
              <span className={isPositive ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.open, selectedPair)}
              </span>
            </div>
            <div className="flex items-center gap-0.5">
              <span className="text-gray-500">H</span>
              <span className={isPositive ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.high, selectedPair)}
              </span>
            </div>
            <div className="flex items-center gap-0.5">
              <span className="text-gray-500">L</span>
              <span className={isPositive ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.low, selectedPair)}
              </span>
            </div>
            <div className="flex items-center gap-0.5">
              <span className="text-gray-500">C</span>
              <span className={isPositive ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.close, selectedPair)}
              </span>
            </div>
            <div className="flex items-center gap-0.5">
              <span className={isPositive ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {isPositive ? '+' : ''}{displayData.change?.toFixed(2)} ({isPositive ? '+' : ''}{displayData.changePercent?.toFixed(2)}%)
              </span>
            </div>
          </>
        )}
      </div>

      {/* Indicator Labels */}
      {(showMA7 || showMA20 || showMA50 || showBB) && (
        <div className="absolute top-6 left-2 z-20 flex flex-wrap items-center gap-2 text-[10px]">
          {showMA7 && (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-0.5" style={{ backgroundColor: '#F7931A' }}></span>
              <span style={{ color: '#F7931A' }}>MA(7)</span>
            </span>
          )}
          {showMA20 && (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-0.5" style={{ backgroundColor: '#2962FF' }}></span>
              <span style={{ color: '#2962FF' }}>MA(20)</span>
            </span>
          )}
          {showMA50 && (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-0.5" style={{ backgroundColor: '#7B1FA2' }}></span>
              <span style={{ color: '#7B1FA2' }}>MA(50)</span>
            </span>
          )}
          {showBB && (
            <span className="flex items-center gap-1">
              <span className="w-2.5 h-0.5" style={{ backgroundColor: '#E91E63' }}></span>
              <span style={{ color: '#E91E63' }}>BOLL(20,2)</span>
            </span>
          )}
        </div>
      )}

      {/* Volume Label */}
      {showVolume && displayData && (
        <div className="absolute bottom-[85px] left-2 z-20 text-[10px] flex items-center gap-1">
          <span className="text-gray-500">Vol</span>
          <span className="text-gray-300">{formatVolume(displayData.volume)}</span>
        </div>
      )}

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: '550px' }}
      />
    </div>
  );
};

export default TradingViewChart;
