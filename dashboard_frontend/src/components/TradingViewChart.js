import { useEffect, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { useTradingContext } from '../context/TradingContext';

const TradingViewChart = () => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const ma7SeriesRef = useRef(null);
  const ma20SeriesRef = useRef(null);
  const ma50SeriesRef = useRef(null);
  const bbUpperSeriesRef = useRef(null);
  const bbLowerSeriesRef = useRef(null);
  const priceLineRef = useRef(null);

  const {
    chartData,
    isDarkMode,
    showMA7,
    showMA20,
    showMA50,
    showBB,
    showVolume,
    stats,
  } = useTradingContext();

  // Initialize chart only once
  useEffect(() => {
    if (!chartContainerRef.current || chartRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { type: 'solid', color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1e222d' },
        horzLines: { color: '#1e222d' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#758696',
          width: 1,
          style: 2,
          labelBackgroundColor: '#2a2e39',
        },
        horzLine: {
          color: '#758696',
          width: 1,
          style: 2,
          labelBackgroundColor: '#2a2e39',
        },
      },
      rightPriceScale: {
        borderColor: '#2a2e39',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: '#2a2e39',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    candlestickSeriesRef.current = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    });

    // Create volume series
    volumeSeriesRef.current = chart.addHistogramSeries({
      color: 'rgba(38, 166, 154, 0.5)',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Create MA series
    ma7SeriesRef.current = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    ma20SeriesRef.current = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    ma50SeriesRef.current = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Create Bollinger Bands series
    bbUpperSeriesRef.current = chart.addLineSeries({
      color: '#9333ea',
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    bbLowerSeriesRef.current = chart.addLineSeries({
      color: '#9333ea',
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, []);

  // Update theme
  useEffect(() => {
    if (!chartRef.current) return;

    const bgColor = isDarkMode ? '#131722' : '#ffffff';
    const textColor = isDarkMode ? '#d1d4dc' : '#131722';
    const gridColor = isDarkMode ? '#1e222d' : '#f0f0f0';
    const borderColor = isDarkMode ? '#2a2e39' : '#e0e0e0';

    chartRef.current.applyOptions({
      layout: {
        background: { type: 'solid', color: bgColor },
        textColor: textColor,
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      rightPriceScale: { borderColor: borderColor },
      timeScale: { borderColor: borderColor },
    });
  }, [isDarkMode]);

  // Helper function to parse timestamp
  const parseTimestamp = (d) => {
    if (typeof d.time === 'string') {
      return Math.floor(new Date(d.time).getTime() / 1000);
    } else if (d.timestamp) {
      return Math.floor(d.timestamp / 1000);
    }
    return Math.floor(d.time / 1000);
  };

  // Update chart data
  useEffect(() => {
    if (!chartData || chartData.length === 0) return;
    if (!candlestickSeriesRef.current) return;

    // Format candlestick data
    const candleData = chartData
      .map((d) => ({
        time: parseTimestamp(d),
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }))
      .filter((d) => !isNaN(d.time) && d.time > 0)
      .sort((a, b) => a.time - b.time);

    // Format volume data
    const volumeData = chartData
      .map((d) => ({
        time: parseTimestamp(d),
        value: d.volume || 0,
        color: d.close >= d.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      }))
      .filter((d) => !isNaN(d.time) && d.time > 0)
      .sort((a, b) => a.time - b.time);

    candlestickSeriesRef.current.setData(candleData);

    if (showVolume && volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(volumeData);
    } else if (volumeSeriesRef.current) {
      volumeSeriesRef.current.setData([]);
    }

    // MA7
    if (showMA7 && ma7SeriesRef.current) {
      const ma7Data = chartData
        .filter((d) => d.ma7 != null)
        .map((d) => ({ time: parseTimestamp(d), value: d.ma7 }))
        .filter((d) => !isNaN(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);
      ma7SeriesRef.current.setData(ma7Data);
    } else if (ma7SeriesRef.current) {
      ma7SeriesRef.current.setData([]);
    }

    // MA20
    if (showMA20 && ma20SeriesRef.current) {
      const ma20Data = chartData
        .filter((d) => d.ma20 != null)
        .map((d) => ({ time: parseTimestamp(d), value: d.ma20 }))
        .filter((d) => !isNaN(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);
      ma20SeriesRef.current.setData(ma20Data);
    } else if (ma20SeriesRef.current) {
      ma20SeriesRef.current.setData([]);
    }

    // MA50
    if (showMA50 && ma50SeriesRef.current) {
      const ma50Data = chartData
        .filter((d) => d.ma50 != null)
        .map((d) => ({ time: parseTimestamp(d), value: d.ma50 }))
        .filter((d) => !isNaN(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);
      ma50SeriesRef.current.setData(ma50Data);
    } else if (ma50SeriesRef.current) {
      ma50SeriesRef.current.setData([]);
    }

    // Bollinger Bands
    if (showBB && bbUpperSeriesRef.current && bbLowerSeriesRef.current) {
      const bbUpperData = chartData
        .filter((d) => d.bb_upper != null)
        .map((d) => ({ time: parseTimestamp(d), value: d.bb_upper }))
        .filter((d) => !isNaN(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);

      const bbLowerData = chartData
        .filter((d) => d.bb_lower != null)
        .map((d) => ({ time: parseTimestamp(d), value: d.bb_lower }))
        .filter((d) => !isNaN(d.time) && d.time > 0)
        .sort((a, b) => a.time - b.time);

      bbUpperSeriesRef.current.setData(bbUpperData);
      bbLowerSeriesRef.current.setData(bbLowerData);
    } else {
      if (bbUpperSeriesRef.current) bbUpperSeriesRef.current.setData([]);
      if (bbLowerSeriesRef.current) bbLowerSeriesRef.current.setData([]);
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [chartData, showMA7, showMA20, showMA50, showBB, showVolume]);

  // Update price line
  useEffect(() => {
    if (!candlestickSeriesRef.current || !stats?.latest_price) return;

    // Remove existing price line
    if (priceLineRef.current) {
      try {
        candlestickSeriesRef.current.removePriceLine(priceLineRef.current);
      } catch (e) {
        // Ignore if already removed
      }
    }

    // Create new price line
    priceLineRef.current = candlestickSeriesRef.current.createPriceLine({
      price: stats.latest_price,
      color: stats.change_24h_percent >= 0 ? '#26a69a' : '#ef5350',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'Current',
    });
  }, [stats?.latest_price, stats?.change_24h_percent]);

  return (
    <div className="relative w-full">
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: '600px' }}
      />

      {/* Legend */}
      <div className="absolute top-2 left-2 flex flex-wrap gap-3 text-xs z-10">
        {showMA7 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-amber-500"></span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA7</span>
          </span>
        )}
        {showMA20 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-blue-500"></span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA20</span>
          </span>
        )}
        {showMA50 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-purple-500"></span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA50</span>
          </span>
        )}
        {showBB && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-purple-600"></span>
            <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>BB</span>
          </span>
        )}
      </div>
    </div>
  );
};

export default TradingViewChart;
