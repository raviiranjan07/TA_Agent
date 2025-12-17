import { useEffect, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { useTradingContext } from '../context/TradingContext';

const TradingViewChart = () => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef({});
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
        textColor: '#787B86',
        fontSize: 12,
        fontFamily: "'Trebuchet MS', Roboto, Ubuntu, sans-serif",
      },
      grid: {
        vertLines: {
          color: '#1e222d',
          visible: true,
        },
        horzLines: {
          color: '#1e222d',
          visible: true,
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: '#758696',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: '#2962FF',
        },
        horzLine: {
          color: '#758696',
          width: 1,
          style: 0,
          visible: true,
          labelVisible: true,
          labelBackgroundColor: '#2962FF',
        },
      },
      rightPriceScale: {
        borderColor: '#2A2E39',
        borderVisible: true,
        scaleMargins: { top: 0.1, bottom: 0.2 },
        autoScale: true,
        alignLabels: true,
      },
      timeScale: {
        borderColor: '#2A2E39',
        borderVisible: true,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
        barSpacing: 6,
        minBarSpacing: 2,
        fixLeftEdge: false,
        fixRightEdge: false,
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

    // Candlestick series
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
        precision: 2,
        minMove: 0.01,
      },
    });

    // Volume series
    seriesRef.current.volume = chart.addHistogramSeries({
      color: 'rgba(38, 166, 154, 0.5)',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // Configure volume price scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
      visible: false,
    });

    // MA7 line series
    seriesRef.current.ma7 = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    // MA20 line series
    seriesRef.current.ma20 = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    // MA50 line series
    seriesRef.current.ma50 = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 1,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    // Bollinger Bands
    seriesRef.current.bbUpper = chart.addLineSeries({
      color: '#9333ea',
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
    });

    seriesRef.current.bbLower = chart.addLineSeries({
      color: '#9333ea',
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
      visible: false,
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
        seriesRef.current = {};
      }
    };
  }, []);

  // Update theme
  useEffect(() => {
    if (!chartRef.current) return;

    const bgColor = isDarkMode ? '#131722' : '#ffffff';
    const textColor = isDarkMode ? '#787B86' : '#131722';
    const gridColor = isDarkMode ? '#1e222d' : '#f0f3fa';
    const borderColor = isDarkMode ? '#2A2E39' : '#e1e3eb';

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
    if (!seriesRef.current.candlestick) return;

    // Format candlestick data
    const candleData = chartData
      .map((d) => ({
        time: parseTimestamp(d),
        open: parseFloat(d.open),
        high: parseFloat(d.high),
        low: parseFloat(d.low),
        close: parseFloat(d.close),
      }))
      .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.open))
      .sort((a, b) => a.time - b.time);

    // Remove duplicates by time
    const uniqueCandleData = candleData.filter((item, index, self) =>
      index === self.findIndex((t) => t.time === item.time)
    );

    // Format volume data
    const volumeData = chartData
      .map((d) => ({
        time: parseTimestamp(d),
        value: parseFloat(d.volume) || 0,
        color: parseFloat(d.close) >= parseFloat(d.open)
          ? 'rgba(38, 166, 154, 0.5)'
          : 'rgba(239, 83, 80, 0.5)',
      }))
      .filter((d) => !isNaN(d.time) && d.time > 0)
      .sort((a, b) => a.time - b.time);

    // Remove duplicates
    const uniqueVolumeData = volumeData.filter((item, index, self) =>
      index === self.findIndex((t) => t.time === item.time)
    );

    // Set candlestick data
    seriesRef.current.candlestick.setData(uniqueCandleData);

    // Volume visibility
    if (showVolume && seriesRef.current.volume) {
      seriesRef.current.volume.setData(uniqueVolumeData);
      seriesRef.current.volume.applyOptions({ visible: true });
    } else if (seriesRef.current.volume) {
      seriesRef.current.volume.setData([]);
      seriesRef.current.volume.applyOptions({ visible: false });
    }

    // MA7
    if (seriesRef.current.ma7) {
      if (showMA7) {
        const ma7Data = chartData
          .filter((d) => d.ma7 != null && !isNaN(parseFloat(d.ma7)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.ma7) }))
          .filter((d) => !isNaN(d.time) && d.time > 0)
          .sort((a, b) => a.time - b.time);
        const uniqueMA7 = ma7Data.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        seriesRef.current.ma7.setData(uniqueMA7);
        seriesRef.current.ma7.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma7.setData([]);
        seriesRef.current.ma7.applyOptions({ visible: false });
      }
    }

    // MA20
    if (seriesRef.current.ma20) {
      if (showMA20) {
        const ma20Data = chartData
          .filter((d) => d.ma20 != null && !isNaN(parseFloat(d.ma20)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.ma20) }))
          .filter((d) => !isNaN(d.time) && d.time > 0)
          .sort((a, b) => a.time - b.time);
        const uniqueMA20 = ma20Data.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        seriesRef.current.ma20.setData(uniqueMA20);
        seriesRef.current.ma20.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma20.setData([]);
        seriesRef.current.ma20.applyOptions({ visible: false });
      }
    }

    // MA50
    if (seriesRef.current.ma50) {
      if (showMA50) {
        const ma50Data = chartData
          .filter((d) => d.ma50 != null && !isNaN(parseFloat(d.ma50)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.ma50) }))
          .filter((d) => !isNaN(d.time) && d.time > 0)
          .sort((a, b) => a.time - b.time);
        const uniqueMA50 = ma50Data.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        seriesRef.current.ma50.setData(uniqueMA50);
        seriesRef.current.ma50.applyOptions({ visible: true });
      } else {
        seriesRef.current.ma50.setData([]);
        seriesRef.current.ma50.applyOptions({ visible: false });
      }
    }

    // Bollinger Bands
    if (seriesRef.current.bbUpper && seriesRef.current.bbLower) {
      if (showBB) {
        const bbUpperData = chartData
          .filter((d) => d.bb_upper != null && !isNaN(parseFloat(d.bb_upper)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.bb_upper) }))
          .filter((d) => !isNaN(d.time) && d.time > 0)
          .sort((a, b) => a.time - b.time);

        const bbLowerData = chartData
          .filter((d) => d.bb_lower != null && !isNaN(parseFloat(d.bb_lower)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.bb_lower) }))
          .filter((d) => !isNaN(d.time) && d.time > 0)
          .sort((a, b) => a.time - b.time);

        const uniqueBBUpper = bbUpperData.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        const uniqueBBLower = bbLowerData.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );

        seriesRef.current.bbUpper.setData(uniqueBBUpper);
        seriesRef.current.bbLower.setData(uniqueBBLower);
        seriesRef.current.bbUpper.applyOptions({ visible: true });
        seriesRef.current.bbLower.applyOptions({ visible: true });
      } else {
        seriesRef.current.bbUpper.setData([]);
        seriesRef.current.bbLower.setData([]);
        seriesRef.current.bbUpper.applyOptions({ visible: false });
        seriesRef.current.bbLower.applyOptions({ visible: false });
      }
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [chartData, showMA7, showMA20, showMA50, showBB, showVolume]);

  // Update price line
  useEffect(() => {
    if (!seriesRef.current.candlestick || !stats?.latest_price) return;

    // Remove existing price line
    if (priceLineRef.current) {
      try {
        seriesRef.current.candlestick.removePriceLine(priceLineRef.current);
      } catch (e) {
        // Ignore if already removed
      }
    }

    // Create new price line
    priceLineRef.current = seriesRef.current.candlestick.createPriceLine({
      price: stats.latest_price,
      color: stats.change_24h_percent >= 0 ? '#26a69a' : '#ef5350',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: '',
    });
  }, [stats?.latest_price, stats?.change_24h_percent]);

  return (
    <div className="relative w-full">
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: '600px' }}
      />

      {/* Legend - Only show when indicators are active */}
      {(showMA7 || showMA20 || showMA50 || showBB) && (
        <div className="absolute top-2 left-2 flex flex-wrap gap-3 text-xs z-10 bg-opacity-80 rounded px-2 py-1"
             style={{ backgroundColor: isDarkMode ? 'rgba(19, 23, 34, 0.8)' : 'rgba(255, 255, 255, 0.8)' }}>
          {showMA7 && (
            <span className="flex items-center gap-1">
              <span className="w-4 h-0.5 bg-amber-500"></span>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA(7)</span>
            </span>
          )}
          {showMA20 && (
            <span className="flex items-center gap-1">
              <span className="w-4 h-0.5 bg-blue-500"></span>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA(20)</span>
            </span>
          )}
          {showMA50 && (
            <span className="flex items-center gap-1">
              <span className="w-4 h-0.5 bg-purple-500"></span>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>MA(50)</span>
            </span>
          )}
          {showBB && (
            <span className="flex items-center gap-1">
              <span className="w-4 h-0.5 bg-purple-600"></span>
              <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>BOLL</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default TradingViewChart;
