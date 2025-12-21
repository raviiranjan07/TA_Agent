import { useEffect, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import { useTradingContext } from '../context/TradingContext';

const TradingViewChart = () => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef({});
  const priceLineRef = useRef(null);

  // State for crosshair OHLCV display
  const [crosshairData, setCrosshairData] = useState(null);

  const {
    chartData,
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

  // Format number with commas
  const formatPrice = (price) => {
    if (price == null || isNaN(price)) return '—';
    return parseFloat(price).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  };

  // Format volume
  const formatVolume = (vol) => {
    if (vol == null || isNaN(vol)) return '—';
    const num = parseFloat(vol);
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
  };

  // Initialize chart only once
  useEffect(() => {
    if (!chartContainerRef.current || chartRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { type: 'solid', color: '#131722' },
        textColor: '#d1d4dc',
        fontSize: 12,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif",
      },
      grid: {
        vertLines: {
          color: 'rgba(42, 46, 57, 0.6)',
          visible: true,
        },
        horzLines: {
          color: 'rgba(42, 46, 57, 0.6)',
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
        borderColor: 'rgba(42, 46, 57, 1)',
        borderVisible: true,
        scaleMargins: { top: 0.1, bottom: 0.2 },
        autoScale: true,
        alignLabels: true,
        entireTextOnly: true,
      },
      timeScale: {
        borderColor: 'rgba(42, 46, 57, 1)',
        borderVisible: true,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 4,
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
      watermark: {
        visible: true,
        fontSize: 52,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(42, 46, 57, 0.5)',
        text: selectedPair || 'BTCUSDT',
      },
    });

    chartRef.current = chart;

    // Candlestick series - TradingView exact colors
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

    // Volume series - at bottom overlay
    seriesRef.current.volume = chart.addHistogramSeries({
      color: 'rgba(38, 166, 154, 0.5)',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    // Configure volume price scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      borderVisible: false,
      visible: false,
    });

    // MA7 line series - Yellow/Amber
    seriesRef.current.ma7 = chart.addLineSeries({
      color: '#F7931A',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // MA20 line series - Blue
    seriesRef.current.ma20 = chart.addLineSeries({
      color: '#2962FF',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // MA50 line series - Purple
    seriesRef.current.ma50 = chart.addLineSeries({
      color: '#7B1FA2',
      lineWidth: 1,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Bollinger Bands - Upper
    seriesRef.current.bbUpper = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 0,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Bollinger Bands - Lower
    seriesRef.current.bbLower = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 0,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Bollinger Bands - Middle (SMA 20)
    seriesRef.current.bbMiddle = chart.addLineSeries({
      color: '#E91E63',
      lineWidth: 1,
      lineStyle: 2,
      crosshairMarkerVisible: false,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Subscribe to crosshair move for OHLCV display
    chart.subscribeCrosshairMove((param) => {
      if (!param || !param.time || !param.seriesData) {
        setCrosshairData(null);
        return;
      }

      const candleData = param.seriesData.get(seriesRef.current.candlestick);
      const volumeData = param.seriesData.get(seriesRef.current.volume);

      if (candleData) {
        setCrosshairData({
          time: param.time,
          open: candleData.open,
          high: candleData.high,
          low: candleData.low,
          close: candleData.close,
          volume: volumeData?.value || 0,
          isUp: candleData.close >= candleData.open,
        });
      }
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

  // Update watermark when pair changes
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.applyOptions({
      watermark: {
        visible: true,
        fontSize: 52,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(42, 46, 57, 0.5)',
        text: selectedPair || 'BTCUSDT',
      },
    });
  }, [selectedPair]);

  // Update theme
  useEffect(() => {
    if (!chartRef.current) return;

    const bgColor = isDarkMode ? '#131722' : '#ffffff';
    const textColor = isDarkMode ? '#d1d4dc' : '#131722';
    const gridColor = isDarkMode ? 'rgba(42, 46, 57, 0.6)' : 'rgba(42, 46, 57, 0.2)';
    const borderColor = isDarkMode ? 'rgba(42, 46, 57, 1)' : '#e1e3eb';
    const watermarkColor = isDarkMode ? 'rgba(42, 46, 57, 0.5)' : 'rgba(0, 0, 0, 0.06)';

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
      watermark: {
        color: watermarkColor,
      },
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
      .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.open) && !isNaN(d.high) && !isNaN(d.low) && !isNaN(d.close))
      .sort((a, b) => a.time - b.time);

    // Remove duplicates by time
    const uniqueCandleData = candleData.filter((item, index, self) =>
      index === self.findIndex((t) => t.time === item.time)
    );

    // Format volume data with proper colors
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
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
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
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
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
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
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
    if (seriesRef.current.bbUpper && seriesRef.current.bbLower && seriesRef.current.bbMiddle) {
      if (showBB) {
        const bbUpperData = chartData
          .filter((d) => d.bb_upper != null && !isNaN(parseFloat(d.bb_upper)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.bb_upper) }))
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
          .sort((a, b) => a.time - b.time);

        const bbLowerData = chartData
          .filter((d) => d.bb_lower != null && !isNaN(parseFloat(d.bb_lower)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.bb_lower) }))
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
          .sort((a, b) => a.time - b.time);

        const bbMiddleData = chartData
          .filter((d) => d.bb_middle != null && !isNaN(parseFloat(d.bb_middle)))
          .map((d) => ({ time: parseTimestamp(d), value: parseFloat(d.bb_middle) }))
          .filter((d) => !isNaN(d.time) && d.time > 0 && !isNaN(d.value))
          .sort((a, b) => a.time - b.time);

        const uniqueBBUpper = bbUpperData.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        const uniqueBBLower = bbLowerData.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );
        const uniqueBBMiddle = bbMiddleData.filter((item, index, self) =>
          index === self.findIndex((t) => t.time === item.time)
        );

        seriesRef.current.bbUpper.setData(uniqueBBUpper);
        seriesRef.current.bbLower.setData(uniqueBBLower);
        seriesRef.current.bbMiddle.setData(uniqueBBMiddle);
        seriesRef.current.bbUpper.applyOptions({ visible: true });
        seriesRef.current.bbLower.applyOptions({ visible: true });
        seriesRef.current.bbMiddle.applyOptions({ visible: true });
      } else {
        seriesRef.current.bbUpper.setData([]);
        seriesRef.current.bbLower.setData([]);
        seriesRef.current.bbMiddle.setData([]);
        seriesRef.current.bbUpper.applyOptions({ visible: false });
        seriesRef.current.bbLower.applyOptions({ visible: false });
        seriesRef.current.bbMiddle.applyOptions({ visible: false });
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

  // Get latest data for display when no crosshair
  const latestData = chartData && chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const displayData = crosshairData || (latestData ? {
    open: parseFloat(latestData.open),
    high: parseFloat(latestData.high),
    low: parseFloat(latestData.low),
    close: parseFloat(latestData.close),
    volume: parseFloat(latestData.volume),
    isUp: parseFloat(latestData.close) >= parseFloat(latestData.open),
  } : null);

  return (
    <div className="relative w-full">
      {/* OHLCV Legend - TradingView Style */}
      <div
        className="absolute top-2 left-3 z-10 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs font-mono"
        style={{ backgroundColor: 'transparent' }}
      >
        {/* Symbol and Timeframe */}
        <div className="flex items-center gap-2">
          <span className={`font-semibold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
            {selectedPair}
          </span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] ${isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>
            {selectedTimeframe}
          </span>
        </div>

        {/* OHLC Values */}
        {displayData && (
          <>
            <div className="flex items-center gap-1">
              <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>O</span>
              <span className={displayData.isUp ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.open)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>H</span>
              <span className={displayData.isUp ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.high)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>L</span>
              <span className={displayData.isUp ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.low)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>C</span>
              <span className={displayData.isUp ? 'text-[#26a69a]' : 'text-[#ef5350]'}>
                {formatPrice(displayData.close)}
              </span>
            </div>
            {showVolume && (
              <div className="flex items-center gap-1">
                <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>Vol</span>
                <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  {formatVolume(displayData.volume)}
                </span>
              </div>
            )}
          </>
        )}
      </div>

      {/* Indicator Legend */}
      {(showMA7 || showMA20 || showMA50 || showBB) && (
        <div
          className="absolute top-8 left-3 z-10 flex flex-wrap items-center gap-3 text-[11px]"
          style={{ backgroundColor: 'transparent' }}
        >
          {showMA7 && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5" style={{ backgroundColor: '#F7931A' }}></span>
              <span style={{ color: '#F7931A' }}>MA(7)</span>
            </span>
          )}
          {showMA20 && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5" style={{ backgroundColor: '#2962FF' }}></span>
              <span style={{ color: '#2962FF' }}>MA(20)</span>
            </span>
          )}
          {showMA50 && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5" style={{ backgroundColor: '#7B1FA2' }}></span>
              <span style={{ color: '#7B1FA2' }}>MA(50)</span>
            </span>
          )}
          {showBB && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5" style={{ backgroundColor: '#E91E63' }}></span>
              <span style={{ color: '#E91E63' }}>BOLL(20,2)</span>
            </span>
          )}
        </div>
      )}

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: '600px' }}
      />
    </div>
  );
};

export default TradingViewChart;
