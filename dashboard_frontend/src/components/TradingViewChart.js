import React, { useEffect, useRef, useCallback } from 'react';
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
  const resizeObserverRef = useRef(null);

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

  // Chart colors
  const colors = {
    backgroundColor: isDarkMode ? '#131722' : '#ffffff',
    textColor: isDarkMode ? '#d1d4dc' : '#131722',
    gridColor: isDarkMode ? '#1e222d' : '#f0f0f0',
    borderColor: isDarkMode ? '#2a2e39' : '#e0e0e0',
    upColor: '#26a69a',
    downColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
    volumeUpColor: 'rgba(38, 166, 154, 0.5)',
    volumeDownColor: 'rgba(239, 83, 80, 0.5)',
    ma7Color: '#f59e0b',
    ma20Color: '#3b82f6',
    ma50Color: '#8b5cf6',
    bbColor: '#9333ea',
    crosshairColor: isDarkMode ? '#758696' : '#9B9B9B',
  };

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { type: 'solid', color: colors.backgroundColor },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: colors.crosshairColor,
          width: 1,
          style: 2,
          labelBackgroundColor: colors.borderColor,
        },
        horzLine: {
          color: colors.crosshairColor,
          width: 1,
          style: 2,
          labelBackgroundColor: colors.borderColor,
        },
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: colors.borderColor,
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time) => {
          const date = new Date(time * 1000);
          return date.toLocaleString('en-IN', {
            timeZone: 'Asia/Kolkata',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          });
        },
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
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: colors.upColor,
      downColor: colors.downColor,
      borderDownColor: colors.downColor,
      borderUpColor: colors.upColor,
      wickDownColor: colors.wickDownColor,
      wickUpColor: colors.wickUpColor,
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: colors.volumeUpColor,
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.85,
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // Create MA7 line series
    const ma7Series = chart.addLineSeries({
      color: colors.ma7Color,
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    ma7SeriesRef.current = ma7Series;

    // Create MA20 line series
    const ma20Series = chart.addLineSeries({
      color: colors.ma20Color,
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    ma20SeriesRef.current = ma20Series;

    // Create MA50 line series
    const ma50Series = chart.addLineSeries({
      color: colors.ma50Color,
      lineWidth: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    ma50SeriesRef.current = ma50Series;

    // Create Bollinger Bands series
    const bbUpperSeries = chart.addLineSeries({
      color: colors.bbColor,
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    bbUpperSeriesRef.current = bbUpperSeries;

    const bbLowerSeries = chart.addLineSeries({
      color: colors.bbColor,
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    bbLowerSeriesRef.current = bbLowerSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    resizeObserverRef.current = new ResizeObserver(handleResize);
    resizeObserverRef.current.observe(chartContainerRef.current);

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      }
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, []);

  // Update chart theme
  useEffect(() => {
    if (!chartRef.current) return;

    chartRef.current.applyOptions({
      layout: {
        background: { type: 'solid', color: colors.backgroundColor },
        textColor: colors.textColor,
      },
      grid: {
        vertLines: { color: colors.gridColor },
        horzLines: { color: colors.gridColor },
      },
      rightPriceScale: {
        borderColor: colors.borderColor,
      },
      timeScale: {
        borderColor: colors.borderColor,
      },
    });
  }, [isDarkMode, colors.backgroundColor, colors.textColor, colors.gridColor, colors.borderColor]);

  // Update chart data
  useEffect(() => {
    if (!chartData || chartData.length === 0) return;
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;

    // Format data for lightweight-charts
    const candleData = chartData.map((d) => {
      // Parse time from ISO string or timestamp
      let timestamp;
      if (typeof d.time === 'string') {
        timestamp = Math.floor(new Date(d.time).getTime() / 1000);
      } else if (d.timestamp) {
        timestamp = Math.floor(d.timestamp / 1000);
      } else {
        timestamp = Math.floor(d.time / 1000);
      }

      return {
        time: timestamp,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      };
    }).filter(d => !isNaN(d.time) && d.time > 0);

    const volumeData = chartData.map((d, index) => {
      let timestamp;
      if (typeof d.time === 'string') {
        timestamp = Math.floor(new Date(d.time).getTime() / 1000);
      } else if (d.timestamp) {
        timestamp = Math.floor(d.timestamp / 1000);
      } else {
        timestamp = Math.floor(d.time / 1000);
      }

      return {
        time: timestamp,
        value: d.volume || 0,
        color: d.close >= d.open ? colors.volumeUpColor : colors.volumeDownColor,
      };
    }).filter(d => !isNaN(d.time) && d.time > 0);

    // Set candlestick data
    candlestickSeriesRef.current.setData(candleData);

    // Set volume data
    if (showVolume) {
      volumeSeriesRef.current.setData(volumeData);
    } else {
      volumeSeriesRef.current.setData([]);
    }

    // MA7 data
    if (showMA7 && ma7SeriesRef.current) {
      const ma7Data = chartData
        .filter(d => d.ma7 != null)
        .map((d) => {
          let timestamp;
          if (typeof d.time === 'string') {
            timestamp = Math.floor(new Date(d.time).getTime() / 1000);
          } else if (d.timestamp) {
            timestamp = Math.floor(d.timestamp / 1000);
          } else {
            timestamp = Math.floor(d.time / 1000);
          }
          return { time: timestamp, value: d.ma7 };
        })
        .filter(d => !isNaN(d.time) && d.time > 0);
      ma7SeriesRef.current.setData(ma7Data);
    } else if (ma7SeriesRef.current) {
      ma7SeriesRef.current.setData([]);
    }

    // MA20 data
    if (showMA20 && ma20SeriesRef.current) {
      const ma20Data = chartData
        .filter(d => d.ma20 != null)
        .map((d) => {
          let timestamp;
          if (typeof d.time === 'string') {
            timestamp = Math.floor(new Date(d.time).getTime() / 1000);
          } else if (d.timestamp) {
            timestamp = Math.floor(d.timestamp / 1000);
          } else {
            timestamp = Math.floor(d.time / 1000);
          }
          return { time: timestamp, value: d.ma20 };
        })
        .filter(d => !isNaN(d.time) && d.time > 0);
      ma20SeriesRef.current.setData(ma20Data);
    } else if (ma20SeriesRef.current) {
      ma20SeriesRef.current.setData([]);
    }

    // MA50 data
    if (showMA50 && ma50SeriesRef.current) {
      const ma50Data = chartData
        .filter(d => d.ma50 != null)
        .map((d) => {
          let timestamp;
          if (typeof d.time === 'string') {
            timestamp = Math.floor(new Date(d.time).getTime() / 1000);
          } else if (d.timestamp) {
            timestamp = Math.floor(d.timestamp / 1000);
          } else {
            timestamp = Math.floor(d.time / 1000);
          }
          return { time: timestamp, value: d.ma50 };
        })
        .filter(d => !isNaN(d.time) && d.time > 0);
      ma50SeriesRef.current.setData(ma50Data);
    } else if (ma50SeriesRef.current) {
      ma50SeriesRef.current.setData([]);
    }

    // Bollinger Bands data
    if (showBB && bbUpperSeriesRef.current && bbLowerSeriesRef.current) {
      const bbUpperData = chartData
        .filter(d => d.bb_upper != null)
        .map((d) => {
          let timestamp;
          if (typeof d.time === 'string') {
            timestamp = Math.floor(new Date(d.time).getTime() / 1000);
          } else if (d.timestamp) {
            timestamp = Math.floor(d.timestamp / 1000);
          } else {
            timestamp = Math.floor(d.time / 1000);
          }
          return { time: timestamp, value: d.bb_upper };
        })
        .filter(d => !isNaN(d.time) && d.time > 0);

      const bbLowerData = chartData
        .filter(d => d.bb_lower != null)
        .map((d) => {
          let timestamp;
          if (typeof d.time === 'string') {
            timestamp = Math.floor(new Date(d.time).getTime() / 1000);
          } else if (d.timestamp) {
            timestamp = Math.floor(d.timestamp / 1000);
          } else {
            timestamp = Math.floor(d.time / 1000);
          }
          return { time: timestamp, value: d.bb_lower };
        })
        .filter(d => !isNaN(d.time) && d.time > 0);

      bbUpperSeriesRef.current.setData(bbUpperData);
      bbLowerSeriesRef.current.setData(bbLowerData);
    } else {
      if (bbUpperSeriesRef.current) bbUpperSeriesRef.current.setData([]);
      if (bbLowerSeriesRef.current) bbLowerSeriesRef.current.setData([]);
    }

    // Fit content
    chartRef.current.timeScale().fitContent();

  }, [chartData, showMA7, showMA20, showMA50, showBB, showVolume, colors.volumeUpColor, colors.volumeDownColor]);

  // Add price line for current price
  useEffect(() => {
    if (!candlestickSeriesRef.current || !stats?.latest_price) return;

    // Remove existing price lines
    const priceLine = candlestickSeriesRef.current.createPriceLine({
      price: stats.latest_price,
      color: stats.change_24h_percent >= 0 ? colors.upColor : colors.downColor,
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'Current',
    });

    return () => {
      if (candlestickSeriesRef.current) {
        candlestickSeriesRef.current.removePriceLine(priceLine);
      }
    };
  }, [stats?.latest_price, stats?.change_24h_percent, colors.upColor, colors.downColor]);

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
