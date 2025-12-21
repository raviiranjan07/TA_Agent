import { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { useTradingContext } from '../context/TradingContext';

const TradingViewChart = () => {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const ma7SeriesRef = useRef(null);
  const ma20SeriesRef = useRef(null);
  const ma50SeriesRef = useRef(null);
  const bbUpperRef = useRef(null);
  const bbMiddleRef = useRef(null);
  const bbLowerRef = useRef(null);

  const [currentCandle, setCurrentCandle] = useState(null);

  const {
    chartData,
    indicators,
    showMA7,
    showMA20,
    showMA50,
    showBB,
    showVolume,
    selectedPair,
    selectedTimeframe,
  } = useTradingContext();

  // Format price
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

  // Convert timestamp to seconds
  const toSeconds = (candle) => {
    if (candle.timestamp) {
      return Math.floor(candle.timestamp / 1000);
    }
    if (candle.time) {
      const d = new Date(candle.time);
      return Math.floor(d.getTime() / 1000);
    }
    return 0;
  };

  // Initialize chart once
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { type: 'solid', color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1e222d' },
        horzLines: { color: '#1e222d' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          labelBackgroundColor: '#2962FF',
        },
        horzLine: {
          labelBackgroundColor: '#2962FF',
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
    });

    chartRef.current = chart;

    // Candlestick series
    candleSeriesRef.current = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderUpColor: '#26a69a',
      borderDownColor: '#ef5350',
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    // Volume series
    volumeSeriesRef.current = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    // MA7
    ma7SeriesRef.current = chart.addLineSeries({
      color: '#f7931a',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // MA20
    ma20SeriesRef.current = chart.addLineSeries({
      color: '#2962ff',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // MA50
    ma50SeriesRef.current = chart.addLineSeries({
      color: '#7b1fa2',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Bollinger Bands
    bbUpperRef.current = chart.addLineSeries({
      color: '#e91e63',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    bbMiddleRef.current = chart.addLineSeries({
      color: '#e91e63',
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    bbLowerRef.current = chart.addLineSeries({
      color: '#e91e63',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (param.time && param.seriesData) {
        const data = param.seriesData.get(candleSeriesRef.current);
        const volData = param.seriesData.get(volumeSeriesRef.current);
        if (data) {
          setCurrentCandle({
            ...data,
            volume: volData?.value || 0,
          });
        }
      } else {
        setCurrentCandle(null);
      }
    });

    // Resize handler
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, []);

  // Update data when chartData or indicators change
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    if (!chartData || chartData.length === 0) return;

    // Process candle data
    const candles = [];
    const volumes = [];
    const seen = new Set();

    for (const c of chartData) {
      const time = toSeconds(c);
      if (time <= 0 || seen.has(time)) continue;
      seen.add(time);

      const open = parseFloat(c.open);
      const high = parseFloat(c.high);
      const low = parseFloat(c.low);
      const close = parseFloat(c.close);
      const volume = parseFloat(c.volume) || 0;

      if (isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) continue;

      candles.push({ time, open, high, low, close });
      volumes.push({
        time,
        value: volume,
        color: close >= open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)',
      });
    }

    candles.sort((a, b) => a.time - b.time);
    volumes.sort((a, b) => a.time - b.time);

    candleSeriesRef.current.setData(candles);

    // Volume
    if (showVolume) {
      volumeSeriesRef.current.setData(volumes);
    } else {
      volumeSeriesRef.current.setData([]);
    }

    // Use indicators data if available, otherwise use chartData
    const indData = indicators && indicators.length > 0 ? indicators : chartData;

    // MA7
    if (showMA7) {
      const ma7Data = [];
      const seenMa7 = new Set();
      for (const d of indData) {
        if (d.ma7 == null) continue;
        const time = toSeconds(d);
        if (time <= 0 || seenMa7.has(time)) continue;
        seenMa7.add(time);
        const value = parseFloat(d.ma7);
        if (!isNaN(value)) ma7Data.push({ time, value });
      }
      ma7Data.sort((a, b) => a.time - b.time);
      ma7SeriesRef.current.setData(ma7Data);
    } else {
      ma7SeriesRef.current.setData([]);
    }

    // MA20
    if (showMA20) {
      const ma20Data = [];
      const seenMa20 = new Set();
      for (const d of indData) {
        if (d.ma20 == null) continue;
        const time = toSeconds(d);
        if (time <= 0 || seenMa20.has(time)) continue;
        seenMa20.add(time);
        const value = parseFloat(d.ma20);
        if (!isNaN(value)) ma20Data.push({ time, value });
      }
      ma20Data.sort((a, b) => a.time - b.time);
      ma20SeriesRef.current.setData(ma20Data);
    } else {
      ma20SeriesRef.current.setData([]);
    }

    // MA50
    if (showMA50) {
      const ma50Data = [];
      const seenMa50 = new Set();
      for (const d of indData) {
        if (d.ma50 == null) continue;
        const time = toSeconds(d);
        if (time <= 0 || seenMa50.has(time)) continue;
        seenMa50.add(time);
        const value = parseFloat(d.ma50);
        if (!isNaN(value)) ma50Data.push({ time, value });
      }
      ma50Data.sort((a, b) => a.time - b.time);
      ma50SeriesRef.current.setData(ma50Data);
    } else {
      ma50SeriesRef.current.setData([]);
    }

    // Bollinger Bands
    if (showBB) {
      const upper = [];
      const middle = [];
      const lower = [];
      const seenBB = new Set();

      for (const d of indData) {
        const time = toSeconds(d);
        if (time <= 0 || seenBB.has(time)) continue;
        seenBB.add(time);

        if (d.bb_upper != null) {
          const v = parseFloat(d.bb_upper);
          if (!isNaN(v)) upper.push({ time, value: v });
        }
        if (d.bb_middle != null) {
          const v = parseFloat(d.bb_middle);
          if (!isNaN(v)) middle.push({ time, value: v });
        }
        if (d.bb_lower != null) {
          const v = parseFloat(d.bb_lower);
          if (!isNaN(v)) lower.push({ time, value: v });
        }
      }

      upper.sort((a, b) => a.time - b.time);
      middle.sort((a, b) => a.time - b.time);
      lower.sort((a, b) => a.time - b.time);

      bbUpperRef.current.setData(upper);
      bbMiddleRef.current.setData(middle);
      bbLowerRef.current.setData(lower);
    } else {
      bbUpperRef.current.setData([]);
      bbMiddleRef.current.setData([]);
      bbLowerRef.current.setData([]);
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [chartData, indicators, showMA7, showMA20, showMA50, showBB, showVolume]);

  // Get display data
  const lastCandle = chartData && chartData.length > 0 ? chartData[chartData.length - 1] : null;
  const displayCandle = currentCandle || (lastCandle ? {
    open: parseFloat(lastCandle.open),
    high: parseFloat(lastCandle.high),
    low: parseFloat(lastCandle.low),
    close: parseFloat(lastCandle.close),
    volume: parseFloat(lastCandle.volume) || 0,
  } : null);

  const isUp = displayCandle ? displayCandle.close >= displayCandle.open : true;
  const priceColor = isUp ? '#26a69a' : '#ef5350';

  return (
    <div className="relative w-full" style={{ backgroundColor: '#131722' }}>
      {/* OHLCV Header */}
      <div className="absolute top-1 left-2 z-10 flex items-center gap-3 text-xs font-mono">
        <span className="text-white font-bold">{selectedPair}</span>
        <span className="text-gray-500 bg-gray-800 px-1 rounded text-[10px]">
          {selectedTimeframe.toUpperCase()}
        </span>

        {displayCandle && (
          <>
            <span>
              <span className="text-gray-500">O </span>
              <span style={{ color: priceColor }}>{formatPrice(displayCandle.open)}</span>
            </span>
            <span>
              <span className="text-gray-500">H </span>
              <span style={{ color: priceColor }}>{formatPrice(displayCandle.high)}</span>
            </span>
            <span>
              <span className="text-gray-500">L </span>
              <span style={{ color: priceColor }}>{formatPrice(displayCandle.low)}</span>
            </span>
            <span>
              <span className="text-gray-500">C </span>
              <span style={{ color: priceColor }}>{formatPrice(displayCandle.close)}</span>
            </span>
            {showVolume && (
              <span>
                <span className="text-gray-500">Vol </span>
                <span className="text-gray-300">{formatVolume(displayCandle.volume)}</span>
              </span>
            )}
          </>
        )}
      </div>

      {/* Indicator Legend */}
      {(showMA7 || showMA20 || showMA50 || showBB) && (
        <div className="absolute top-5 left-2 z-10 flex items-center gap-2 text-[10px]">
          {showMA7 && (
            <span className="flex items-center gap-0.5">
              <span className="w-2 h-0.5 bg-[#f7931a]"></span>
              <span className="text-[#f7931a]">MA7</span>
            </span>
          )}
          {showMA20 && (
            <span className="flex items-center gap-0.5">
              <span className="w-2 h-0.5 bg-[#2962ff]"></span>
              <span className="text-[#2962ff]">MA20</span>
            </span>
          )}
          {showMA50 && (
            <span className="flex items-center gap-0.5">
              <span className="w-2 h-0.5 bg-[#7b1fa2]"></span>
              <span className="text-[#7b1fa2]">MA50</span>
            </span>
          )}
          {showBB && (
            <span className="flex items-center gap-0.5">
              <span className="w-2 h-0.5 bg-[#e91e63]"></span>
              <span className="text-[#e91e63]">BOLL</span>
            </span>
          )}
        </div>
      )}

      {/* Chart */}
      <div ref={chartContainerRef} style={{ height: '500px' }} />
    </div>
  );
};

export default TradingViewChart;
